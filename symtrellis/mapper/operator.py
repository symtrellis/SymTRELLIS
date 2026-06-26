"""
Linear symmetry operators for sparse voxel features.

This file uses two index spaces:

1. Relation-expanded entry space.
   LinearCoefficient maps src entries to dst entries. Its sizes are num_src
   and num_dst, not the original sparse feature row count N.

2. Original feature row space.
   SymmetryProjector uses rows_src and rows_dst to gather from and scatter back
   to the same original feature tensor feats with shape [N, C].

Data flow:
    feats[N, C]
        -> feats[rows_src]              # [num_src, C]
        -> LinearCoefficient.apply      # [num_dst, C]
        -> index_add by rows_dst        # [N, C]

rows_src and rows_dst may contain duplicates. Duplicates are expected when
multiple symmetry relations use or contribute to the same original voxel row.
"""

from dataclasses import dataclass
from itertools import accumulate
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LinearCoefficient:
    """
    Learned linear map from relation-expanded src entries to dst entries.

    This class does not store coordinates, sample ids, or original feature row
    indices. It only stores the edge-wise coefficients predicted by the mapper.

    Shapes:
      num_src: number of src entries.
      num_dst: number of dst entries.
      feat_dim: feature channel count C.
      e_ids_src: [E], edge -> src entry index in [0, num_src).
      e_ids_dst: [E], edge -> dst entry index in [0, num_dst).
      s: [E, rank], edge-wise low-rank coefficient.
      w: [E], normalized edge aggregation weight.
      Ut: [rank, C].
      V: [C, rank].

    apply(feats_src) maps [num_src, C] to [num_dst, C].
    apply_transposed(feats_dst) maps [num_dst, C] to [num_src, C].
    """

    device: torch.device
    dtype: torch.dtype

    num_src: int
    num_dst: int
    feat_dim: int

    e_ids_dst: torch.Tensor  # [E,] int64
    e_ids_src: torch.Tensor  # [E,] int64
    s: torch.Tensor  # [E, rank]
    w: torch.Tensor  # [E,]
    Ut: torch.Tensor  # [rank, dim]
    V: torch.Tensor  # [dim, rank]

    def to(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> "LinearCoefficient":
        target_dtype = self.dtype if dtype is None else dtype
        return LinearCoefficient(
            device=device,
            dtype=target_dtype,
            num_src=self.num_src,
            num_dst=self.num_dst,
            feat_dim=self.feat_dim,
            e_ids_dst=self.e_ids_dst.to(device=device, non_blocking=True),
            e_ids_src=self.e_ids_src.to(device=device, non_blocking=True),
            s=self.s.to(device=device, dtype=target_dtype, non_blocking=True),
            w=self.w.to(device=device, dtype=target_dtype, non_blocking=True),
            Ut=self.Ut.to(device=device, dtype=target_dtype, non_blocking=True),
            V=self.V.to(device=device, dtype=target_dtype, non_blocking=True),
        )

    def apply(self, feats_src: torch.Tensor) -> torch.Tensor:
        """
        Apply the src-to-dst map.

        Args:
          feats_src: [num_src, C] src entry features.

        Returns:
          out: [num_dst, C] dst entry features.
        """
        assert feats_src.device == self.device
        assert feats_src.dtype == self.dtype
        assert feats_src.shape[1] == self.feat_dim

        # Build edge-wise transformed src features.
        Ufeat = F.linear(feats_src, self.Ut)  # [num_src, rank]
        feat_rot = feats_src[self.e_ids_src] + F.linear(self.s * Ufeat[self.e_ids_src], self.V)  # [E, 8]

        # Aggregate edge contributions onto dst entries.
        out = feats_src.new_zeros((self.num_dst, self.feat_dim))
        out.index_add_(0, self.e_ids_dst, feat_rot * self.w[..., None])

        return out

    def apply_transposed(self, feats_dst: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint dst-to-src map used by conjugate gradient.

        Args:
          feats_dst: [num_dst, C] dst entry features.

        Returns:
          out: [num_src, C] src entry features.
        """
        assert feats_dst.device == self.device
        assert feats_dst.dtype == self.dtype
        assert feats_dst.shape[1] == self.feat_dim

        # Apply the transpose of the low-rank edge transform.
        Vfeat = F.linear(feats_dst, self.V.t())
        feat_rot_T = feats_dst[self.e_ids_dst] + F.linear(self.s * Vfeat[self.e_ids_dst], self.Ut.t())

        # Scatter adjoint edge contributions back to src entries.
        out = feats_dst.new_zeros((self.num_src, self.feat_dim))
        out.index_add_(0, self.e_ids_src, feat_rot_T * self.w[..., None])

        return out


def concat_coeff(coeff_list: List[LinearCoefficient]) -> LinearCoefficient:
    """
    Concatenate coefficients in relation-expanded entry space.

    e_ids_src are offset by cumulative num_src because they index src entries.
    e_ids_dst are offset by cumulative num_dst because they index dst entries.

    This function does not handle original feature row indices. Use concat_rows
    for rows_src and rows_dst.
    """

    assert len(coeff_list) > 0
    c0 = coeff_list[0]

    assert all(c0.feat_dim == coeff.feat_dim for coeff in coeff_list)
    assert all(torch.allclose(c0.Ut, coeff.Ut, rtol=1e-5, atol=1e-8) for coeff in coeff_list)
    assert all(torch.allclose(c0.V, coeff.V, rtol=1e-5, atol=1e-8) for coeff in coeff_list)
    assert all(c0.device == coeff.device for coeff in coeff_list)
    assert all(c0.dtype == coeff.dtype for coeff in coeff_list)

    offsets_src = list(accumulate([coeff.num_src for coeff in coeff_list], initial=0))[:-1]
    offsets_dst = list(accumulate([coeff.num_dst for coeff in coeff_list], initial=0))[:-1]

    num_cat_src = sum(coeff.num_src for coeff in coeff_list)
    num_cat_dst = sum(coeff.num_dst for coeff in coeff_list)

    e_ids_cat_src = torch.cat([coeff_list[i].e_ids_src + offsets_src[i] for i in range(len(coeff_list))])
    e_ids_cat_dst = torch.cat([coeff_list[i].e_ids_dst + offsets_dst[i] for i in range(len(coeff_list))])

    s_cat = torch.cat([coeff_list[i].s for i in range(len(coeff_list))])
    w_cat = torch.cat([coeff_list[i].w for i in range(len(coeff_list))])

    return LinearCoefficient(
        device=c0.device,
        dtype=c0.dtype,
        num_src=num_cat_src,
        num_dst=num_cat_dst,
        feat_dim=c0.feat_dim,
        e_ids_dst=e_ids_cat_dst.contiguous(),
        e_ids_src=e_ids_cat_src.contiguous(),
        s=s_cat.contiguous(),
        w=w_cat.contiguous(),
        Ut=c0.Ut.clone(),
        V=c0.V.clone(),
    )


def concat_rows(
    rows_list_src: List[torch.Tensor],
    rows_list_dst: List[torch.Tensor],
    num_rows_list: List[int],
    sample_ids_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate projection rows in original feature row space.

    For item i:
      rows_list_src[i]: [N_i_src], row ids into the i-th feats tensor.
      rows_list_dst[i]: [N_i_dst], row ids into the i-th feats tensor.
      num_rows_list[i]: N_i, number of rows in the i-th feats tensor.
      sample_ids_list[i]: [N_i], sample ids for the i-th feats tensor.

    rows_src and rows_dst may contain duplicates. Duplicates preserve multiple
    symmetry relations that read from or write to the same original row.

    Returns:
      rows_src: [sum_i N_i_src], row ids into concatenated feats.
      rows_dst: [sum_i N_i_dst], row ids into concatenated feats.
      sample_ids: [sum_i N_i], sample ids for concatenated feats rows.
    """
    row_offsets = list(accumulate(num_rows_list, initial=0))[:-1]

    # Offset rows by original feature row counts.
    rows_src = torch.cat([rows_list_src[i] + row_offsets[i] for i in range(len(rows_list_src))])
    rows_dst = torch.cat([rows_list_dst[i] + row_offsets[i] for i in range(len(rows_list_dst))])

    # Offset sample ids so independently-built batches do not collide.
    sample_offsets = list(accumulate([sample_ids.max() + 1 for sample_ids in sample_ids_list], initial=0))[:-1]
    sample_ids = torch.cat([sample_ids_list[i] + sample_offsets[i] for i in range(len(sample_ids_list))])

    return rows_src.contiguous(), rows_dst.contiguous(), sample_ids.contiguous()


def conjugate_gradient(
    A_func,
    rhs: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 10,
    tol: float = 1e-4,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Solve A x = rhs by conjugate gradient without explicitly materializing A.

    A_func computes A @ x. Convergence uses the relative residual
    ||r||_2 / ||rhs||_2.
    """
    rhs_norm = torch.norm(rhs)
    if rhs_norm < 1e-12:
        return torch.zeros_like(rhs) if x0 is None else x0.clone()

    x = torch.zeros_like(rhs) if x0 is None else x0.clone()

    # Calculate initial objective J(x0) for relative objective decrease metric
    if verbose:
        Ax0 = A_func(x)
        J_0 = 0.5 * torch.sum(x * Ax0) - torch.sum(rhs * x)
        if abs(J_0) < 1e-12:
            J_0 = torch.tensor(1.0, device=rhs.device)  # Prevent division by zero if x0 is 0

    r = rhs - A_func(x)
    p = r.clone()
    rsold = torch.sum(r * r)

    for i in range(max_iter):
        Ap = A_func(p)
        p_Ap = torch.sum(p * Ap)
        alpha = rsold / (p_Ap + 1e-12)

        step = alpha * p
        x = x + step
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)

        r_norm = torch.sqrt(rsnew)
        rel_res = r_norm / rhs_norm

        if verbose:
            step_norm = torch.norm(step)
            x_norm = torch.norm(x)
            rel_step = step_norm / (x_norm + 1e-12)

            obj_decrease = 0.5 * alpha * p_Ap
            rel_obj_dec = obj_decrease / abs(J_0)

            print(f"CG Iter {i:2d} | Rel Res: {rel_res.item():.2e} | Rel Step: {rel_step.item():.2e} | Rel Obj Dec: {rel_obj_dec.item():.2e}")

        if rel_res < tol:
            if verbose:
                print(f"CG converged at iteration {i} with Rel Res {rel_res.item():.2e}")
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


class SymmetryProjector:
    """
    Cached gather/scatter rows for self symmetry projection.

    This projector only handles self-projection: rows_src and rows_dst both
    index the same original feature tensor feats [N, C].

    Args:
      num_rows: N, number of original sparse feature rows.
      rows_src: [num_src], rows gathered from feats before applying coeff.
      rows_dst: [num_dst], rows receiving projected dst contributions.

    rows_src and rows_dst may repeat. Repeated dst rows are summed and then
    averaged by counts_dst. Rows with no dst contribution output zero when
    self_include=False.
    """

    def __init__(
        self,
        num_rows: int,
        rows_src: torch.Tensor,
        rows_dst: torch.Tensor,
    ) -> None:
        self.num_rows = num_rows
        self.rows_src = rows_src
        self.rows_dst = rows_dst
        self.counts_dst = torch.bincount(self.rows_dst, minlength=self.num_rows)

    def forward_project(
        self,
        feats: torch.Tensor,
        coeff: LinearCoefficient,
        self_include: bool = False,
    ) -> torch.Tensor:
        """
        Apply the forward symmetry projection.

        Args:
          feats: [N, C] original sparse feature rows.
          coeff: relation-expanded map with coeff.num_src == len(rows_src) and
            coeff.num_dst == len(rows_dst).
          self_include: if True, include each original row as one extra
            contribution before averaging.

        Returns:
          projected_feats: [N, C].
        """
        # Gather original rows into relation-expanded src entries.
        feats_src = feats[self.rows_src]

        # Map src entries to relation-expanded dst entries.
        feats_dst = coeff.apply(feats_src)

        # Scatter/add dst entries back to original rows.
        sum_feats = torch.zeros_like(feats)
        sum_feats.index_add_(0, self.rows_dst, feats_dst)

        # Average all symmetry contributions per original dst row.
        counts_dst = self.counts_dst.unsqueeze(1)
        if self_include:
            return (sum_feats + feats) / (counts_dst + 1.0)

        return sum_feats / counts_dst.clamp_min(1.0)

    @torch.no_grad()
    def least_square_project(
        self,
        feats: torch.Tensor,
        coeff: LinearCoefficient,
        cg_max_iter: int = 10,
        cg_tol: float = 1e-4,
        verbose: bool = False,
        self_include: bool = False,
    ) -> torch.Tensor:
        """
        Project feats onto the range of the symmetry map by least squares.

        Solves:
            min_c || feats - P(c) ||_2^2

        where P is the forward projection defined by rows_src, coeff,
        rows_dst, and counts_dst.

        Args:
          feats: [N, C] original sparse feature rows.
          coeff: relation-expanded src-to-dst map.

        Returns:
          projected_feats: [N, C].
        """
        counts_dst = self.counts_dst.unsqueeze(1)
        if self_include:
            weight_inv = 1.0 / (counts_dst + 1.0)
        else:
            weight_inv = 1.0 / counts_dst.clamp_min(1.0)

        def P(x: torch.Tensor) -> torch.Tensor:
            # Forward projection: gather rows, apply coeff, scatter back.
            feats_src = x[self.rows_src]
            feats_dst = coeff.apply(feats_src)
            y = x.clone() if self_include else torch.zeros_like(x)
            y.index_add_(0, self.rows_dst, feats_dst)
            return y * weight_inv

        def P_T(y: torch.Tensor) -> torch.Tensor:
            # Adjoint projection used in the normal equation P^T P c = P^T feats.
            y_weighted = y * weight_inv
            feats_dst = y_weighted[self.rows_dst]
            feats_src = coeff.apply_transposed(feats_dst)
            x_new = y_weighted.clone() if self_include else torch.zeros_like(y_weighted)
            x_new.index_add_(0, self.rows_src, feats_src)
            return x_new

        def A_func(x: torch.Tensor) -> torch.Tensor:
            return P_T(P(x))

        coeff_feats = conjugate_gradient(
            A_func,
            P_T(feats),
            x0=feats,
            max_iter=cg_max_iter,
            tol=cg_tol,
            verbose=verbose,
        )

        return P(coeff_feats)

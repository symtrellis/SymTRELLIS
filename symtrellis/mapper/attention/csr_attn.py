from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn

from ..encodings.position import RotaryPositionEmbedder
from ..layers import SparseMultiHeadRMSNorm
from .csr_attn_ext import _C


class SparseCSRAttentionFn(torch.autograd.Function):
    """Autograd bridge for memory-efficient directed CSR attention.

    Forward returns only the attended output to Python callers, but saves the
    forward output and per-row/head log-sum-exp. Backward recomputes edge scores
    and probabilities from q/k/lse instead of saving per-edge score/probability
    tensors.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rowptr: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        out: torch.Tensor
        lse: torch.Tensor
        # CUDA computes softmax(q @ k.T / sqrt(D)) @ v for each CSR row and
        # returns lse so backward can reconstruct probabilities.
        out, lse = _C.sparse_csr_attn_forward(q, k, v, rowptr, col)
        ctx.save_for_backward(q, k, v, rowptr, col, out, lse)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        grad_out: torch.Tensor = grad_outputs[0]
        q, k, v, rowptr, col, out, lse = ctx.saved_tensors
        dq: torch.Tensor
        dk: torch.Tensor
        dv: torch.Tensor
        # Only q/k/v are differentiable. rowptr and col describe the fixed CSR
        # sparsity pattern and receive no gradients.
        dq, dk, dv = _C.sparse_csr_attn_backward(grad_out.contiguous(), q, k, v, rowptr, col, out, lse)
        return dq, dk, dv, None, None


def sparse_csr_attn_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    """Run directed CSR scaled dot-product attention on projected q/k/v.

    Args:
        q: CUDA tensor with shape [Nq, H, D], dtype fp32/fp16/bf16.
        k: CUDA tensor with shape [Nk, H, D], same dtype as q.
        v: CUDA tensor with shape [Nk, H, D], same dtype and shape as k.
        rowptr: CUDA int32 tensor with shape [Nq + 1]. Row i attends to
            `col[rowptr[i] : rowptr[i + 1]]`.
        col: CUDA int32 tensor with shape [E]. Entries index rows of k/v.

    Returns:
        CUDA tensor with shape [Nq, H, D] and the same dtype as q.

    This is the user-facing validation boundary for the CUDA extension. The
    backend assumes q/k/v and CSR tensors already satisfy this contract.
    """

    # Projected q/k/v tensor contract.
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("q must have dtype float32, float16, or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have identical dtype")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [N, H, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shape")
    if q.shape[1:] != k.shape[1:]:
        raise ValueError("q and k/v must have identical H and D")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, or 128")

    # CSR tensor contract. Do not inspect CSR values here; that would
    # synchronize GPU state and belongs in offline/reference validation.
    if not rowptr.is_cuda or not col.is_cuda:
        raise ValueError("rowptr and col must be CUDA tensors")
    if rowptr.dtype != torch.int32 or col.dtype != torch.int32:
        raise TypeError("rowptr and col must have dtype torch.int32")
    if rowptr.ndim != 1:
        raise ValueError("rowptr must have shape [Nq + 1]")
    if col.ndim != 1:
        raise ValueError("col must have shape [E]")
    if rowptr.shape[0] != q.shape[0] + 1:
        raise ValueError("rowptr must have length Nq + 1")

    return cast(
        torch.Tensor,
        SparseCSRAttentionFn.apply(
            # The CUDA kernels assume contiguous storage and do not support
            # arbitrary strides.
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            rowptr.contiguous(),
            col.contiguous(),
        ),
    )


class CSRMultiHeadAttention(nn.Module):
    """Directed CSR sparse multi-head attention over sparse token rows.

    The module owns q/k/v projections, optional RoPE, optional q/k RMSNorm, and
    the output projection. The CUDA backend only consumes projected q/k/v and
    caller-provided CSR tensors.

    `x_feats` is always the query/destination branch. `ctx_feats`, when
    provided, is the key/value/source branch; otherwise the module runs
    self-attention over `x_feats`. The module does not build CSR, sort rows,
    apply windows, or scatter outputs back to a different order.

    Empty CSR rows produce zero backend output before `to_out`. If `to_out` has
    bias, that bias still contributes to the module-level output, matching the
    existing attention projection style.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        ctx_feat_dim: Optional[int] = None,
        qkv_bias: bool = True,
        use_rope: bool = True,
        rope_scale: float = 0.75,
        rope_theta: float = 10000.0,
        qk_rms_norm: bool = False,
    ) -> None:
        super().__init__()

        assert feat_dim % num_heads == 0

        self.feat_dim: int = feat_dim
        self.head_dim: int = feat_dim // num_heads
        self.ctx_feat_dim: int = ctx_feat_dim if ctx_feat_dim is not None else feat_dim
        self.num_heads: int = num_heads
        self.use_rope: bool = use_rope
        self.qk_rms_norm: bool = qk_rms_norm

        if self.head_dim not in (16, 32, 64, 128):
            raise ValueError("head_dim must be one of 16, 32, 64, or 128")

        self.to_q = nn.Linear(feat_dim, feat_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(self.ctx_feat_dim, feat_dim * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(feat_dim, feat_dim)

        if use_rope:
            self.rope = RotaryPositionEmbedder(
                hidden_size=self.head_dim,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
            )

    def forward(
        self,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        x_feats: torch.Tensor,
        x_pos: Optional[torch.Tensor] = None,
        ctx_feats: Optional[torch.Tensor] = None,
        ctx_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply directed CSR attention using caller-provided CSR tensors.

        Args:
            rowptr: Int32 CUDA tensor with shape [Nq + 1].
            col: Int32 CUDA tensor with shape [E]. Entries index key/value rows.
            x_feats: Query feature tensor with shape [Nq, feat_dim].
            x_pos: Query positions with shape [Nq, 3], required when RoPE is used.
            ctx_feats: Optional key/value feature tensor with shape
                [Nk, ctx_feat_dim]. If omitted, self-attention uses `x_feats`.
            ctx_pos: Optional key/value positions with shape [Nk, 3], required
                for cross-attention when RoPE is used.

        Returns:
            Tensor with shape [Nq, feat_dim].
        """

        # Select the key/value branch. In self-attention q/k/v all come from
        # x_feats; in cross-attention k/v come from ctx_feats.
        kv_feats: torch.Tensor = x_feats if ctx_feats is None else ctx_feats

        Nq: int = x_feats.shape[0]
        Nk: int = kv_feats.shape[0]

        # Python owns q/k/v projections; the CUDA backend consumes projected
        # [N, H, D] tensors only.
        q = self.to_q(x_feats)
        kv = self.to_kv(kv_feats)

        q = q.reshape(Nq, self.num_heads, self.head_dim)
        kv = kv.reshape(Nk, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=1)

        if self.use_rope:
            # RoPE is applied to q/k before attention. v is intentionally left
            # unchanged.
            if x_pos is None:
                raise ValueError("x_pos is required when use_rope=True")
            kv_pos: Optional[torch.Tensor] = x_pos if ctx_feats is None else ctx_pos
            if kv_pos is None:
                raise ValueError("ctx_pos is required for cross-attention when use_rope=True")
            q = self.rope(q, x_pos)
            k = self.rope(k, kv_pos)

        if self.qk_rms_norm:
            # q/k RMSNorm is also Python-side preprocessing before the backend.
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # Directed CSR attention over rowptr/col. No scatter-back is performed;
        # caller-provided row order is the output row order.
        out = sparse_csr_attn_backend(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            rowptr,
            col,
        )
        out = out.reshape(Nq, self.feat_dim)
        return self.to_out(out)

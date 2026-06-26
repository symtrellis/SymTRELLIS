import torch
import torch.nn as nn

from .encodings.position import FourierPE
from .operator import LinearCoefficient


class EdgeFeatureHead(nn.Module):
    """
    Build edge features for src-to-dst relation entries.

    Inputs are already in relation-expanded entry space. num_src/num_dst are
    not necessarily the original sparse row count.

    Shapes:
      feats_src: [num_src, C], with C == feat_channels.
      pos_src: [num_src, 3]
      coords_dst: [num_dst, 4], coords_dst[:, 0] indexes relation-level condition.
      feats_dst: [num_dst, C], with C == feat_channels.
      pos_dst: [num_dst, 3]
      condition: [num_relations, condition_dim]
      e_ids_src: [E], edge -> src entry index.
      e_ids_dst: [E], edge -> dst entry index.

    For each edge, the MLP input is:
      feats_dst[e_ids_dst]
      feats_src[e_ids_src]
      feats_dst[e_ids_dst] - feats_src[e_ids_src]
      FourierPE(pos_src[e_ids_src] - pos_dst[e_ids_dst]) if use_geom=True
      pos_src[e_ids_src] - pos_dst[e_ids_dst] if use_geom=True
      ||pos_src[e_ids_src] - pos_dst[e_ids_dst]||^2 if use_geom=True
      condition[coords_dst[e_ids_dst, 0]] if use_cond=True

    The relative position vector points from the dst entry to the src entry.

    Output:
      edge_feat: [E, edge_dim].
    """

    def __init__(
        self,
        feat_channels: int,
        condition_dim: int,
        edge_dim: int,
        hidden: int = 128,
        mlp_depth: int = 2,
        use_geom: bool = True,
        pe_num_bands: int = 6,
        pe_max_freq: float = 0.5,
        use_cond: bool = True,
    ):
        """
        Args:
            feat_channels: Feature width of both src and dst entries.
            condition_dim: Width of each relation-level conditioning row.
            edge_dim: Output edge feature width.
            hidden: Hidden width of the edge MLP.
            mlp_depth: Number of linear layers in the edge MLP.
            use_geom: Whether to append relative geometry features.
            pe_num_bands: Number of Fourier bands for relative position.
            pe_max_freq: Maximum Fourier frequency in cycles per unit.
            use_cond: Whether to append relation-level condition features.
        """
        super().__init__()
        self.use_geom = use_geom
        self.use_cond = use_cond

        self.feat_channels = feat_channels
        self.condition_dim = condition_dim
        self.edge_dim = edge_dim

        in_dim = 3 * feat_channels

        if use_geom:
            in_dim += 3 + 1  # delta(3) + dist2(1)
            in_dim += 2 * 3 * pe_num_bands
            self.pe = FourierPE(
                in_dim=3,
                num_bands=pe_num_bands,
                freq_max=pe_max_freq,
                include_input=False,
            )

        if use_cond:
            in_dim += condition_dim

        layers = []
        d = in_dim
        for _ in range(mlp_depth - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, edge_dim)]

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        feats_src: torch.Tensor,
        pos_src: torch.Tensor,
        coords_dst: torch.Tensor,
        feats_dst: torch.Tensor,
        pos_dst: torch.Tensor,
        condition: torch.Tensor,
        e_ids_dst: torch.Tensor,
        e_ids_src: torch.Tensor,
    ):
        """
        Return one feature vector for each src-to-dst edge.

        Args:
          feats_src: [num_src, feat_channels] source entry features.
          pos_src: [num_src, 3] source entry positions.
          coords_dst: [num_dst, 4] destination entry coordinates. The first
            column indexes rows in `condition`.
          feats_dst: [num_dst, feat_channels] destination entry features.
          pos_dst: [num_dst, 3] destination entry positions.
          condition: [num_relations, condition_dim] relation-level condition.
          e_ids_dst: [E] edge -> destination entry index.
          e_ids_src: [E] edge -> source entry index.

        Returns:
          edge_feat: [E, edge_dim].
        """
        feats_edge_dst = feats_dst[e_ids_dst]
        feats_edge_src = feats_src[e_ids_src]
        feats_edge_delta = feats_edge_dst - feats_edge_src

        feats = [feats_edge_dst, feats_edge_src, feats_edge_delta]

        if self.use_geom:
            # Relative vector from dst entry to its src neighbor.
            delta = pos_src[e_ids_src] - pos_dst[e_ids_dst]  # [E, 3]
            dist2 = (delta**2).sum(dim=-1, keepdim=True)
            pe = self.pe(delta)
            feats += [pe, delta, dist2]

        if self.use_cond:
            # coords_dst[:, 0] is a relation id, not a geometric coordinate.
            feats += [condition[coords_dst[e_ids_dst, 0]]]

        x = torch.cat(feats, dim=-1)  # [E, in_dim]
        edge_feat = self.mlp(x)

        return edge_feat


class LowRankMatrixCoefficientHead(nn.Module):
    """
    Predict a LinearCoefficient from edge features.

    This head does not apply the projection itself. It predicts the low-rank
    edge transform `s` and dst-wise normalized edge weights `w`; the returned
    LinearCoefficient later maps feats_src to feats_dst through
    LinearCoefficient.apply(feats_src).
    """

    def __init__(
        self,
        feat_dim: int = 8,
        edge_feat_dim: int = 64,
        rank: int = 8,
    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        self.rank = rank

        self.Ut = nn.Parameter(torch.empty(rank, feat_dim))
        self.V = nn.Parameter(torch.empty(feat_dim, rank))
        nn.init.orthogonal_(self.Ut)
        nn.init.orthogonal_(self.V)

        self.s_layer = nn.Linear(edge_feat_dim, rank, bias=True)
        self.weight_logit_layer = nn.Linear(edge_feat_dim, 1, bias=False)

    def forward(
        self,
        num_src: int,
        num_dst: int,
        edge_feat: torch.Tensor,
        e_ids_dst: torch.Tensor,
        e_ids_src: torch.Tensor,
    ):
        """
        Predict low-rank edge coefficients and dst-normalized edge weights.

        Args:
          num_src: number of relation-expanded src entries.
          num_dst: number of relation-expanded dst entries.
          edge_feat: [E, edge_feat_dim].
          e_ids_src: [E], edge -> src entry index.
          e_ids_dst: [E], edge -> dst entry index.

        Returns:
          coeff: LinearCoefficient with:
            s: [E, rank], edge-wise low-rank transform coefficients.
            w: [E], softmax-normalized over edges with the same e_ids_dst.
        """
        s = self.s_layer(edge_feat)  # [E, rank]

        l: torch.Tensor = self.weight_logit_layer(edge_feat)[..., 0]  # [E]

        # Compute a numerically stable dst-wise softmax over incoming src edges.
        maxv = l.new_full((num_dst,), -float("inf"))
        maxv.scatter_reduce_(0, e_ids_dst, l, reduce="amax", include_self=True)
        x = (l - maxv[e_ids_dst]).exp()

        denom = x.new_zeros((num_dst,))
        denom.index_add_(0, e_ids_dst, x)
        denom = denom.clamp_min(1e-12)

        w = x / denom[e_ids_dst]  # [E]

        return LinearCoefficient(
            device=edge_feat.device,
            dtype=l.dtype,
            num_src=num_src,
            num_dst=num_dst,
            feat_dim=self.feat_dim,
            e_ids_dst=e_ids_dst,
            e_ids_src=e_ids_src,
            s=s,
            w=w,
            Ut=self.Ut.clone(),
            V=self.V.clone(),
        )

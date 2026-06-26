from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Buffer

from ..geometry.neighbors import lattice_ball_offsets, radius_nbr_edges
from .attention.window_index import build_swin_indices
from .blocks import Swin3DLatentMapperBlock
from .config import Swin3DLatentMapperConfig, Swin3DLatentMapperScale, swin_3d_latent_mapper_config
from .encodings.pose_condition import PoseConditioner
from .encodings.position import FourierPE
from .heads import EdgeFeatureHead, LowRankMatrixCoefficientHead
from .operator import LinearCoefficient


class Swin3DLatentMapper(nn.Module):
    """Swin3D implementation of the Spatial-Transform Latent Mapper.

    Predict a sparse src-to-dst latent mapping from a spatial transform.

    Coordinates are relation-expanded sparse lattice coordinates with shape
    [N, 4]. The first column indexes the transform/condition row; the remaining
    three columns are integer grid coordinates. The destination coordinates are
    transformed into the source coordinate system before window attention and
    neighbor search.

    The returned `LinearCoefficient` maps source latent rows to destination
    latent rows in the input row order.
    """

    def __init__(self, cfg: Swin3DLatentMapperConfig) -> None:
        """
        Args:
            cfg: Mapper configuration.
        """
        super().__init__()
        self.cfg = cfg

        self.pose_conditioner = PoseConditioner(
            condition_dim=cfg.condition_dim,
            t_num_freqs=cfg.t_num_freqs,
            t_include_input=cfg.t_include_input,
            freq_min=cfg.t_freq_min,
            sign_dim=cfg.sign_dim,
            hidden_dim=cfg.condition_hidden_dim,
            num_layers=cfg.condition_num_layers,
        )

        self.pos_pe = FourierPE(
            in_dim=3,
            num_bands=cfg.pos_pe_num_bands,
            freq_max=cfg.pos_pe_freq_max,
            include_input=cfg.pos_pe_include_input,
        )
        self.in_proj = nn.Linear(self.pos_pe.out_dim, cfg.feat_dim)
        self.in_norm = nn.LayerNorm(cfg.feat_dim)

        self.blocks = nn.ModuleList(
            [
                Swin3DLatentMapperBlock(
                    feat_dim=cfg.feat_dim,
                    num_heads=cfg.num_heads,
                    condition_dim=cfg.condition_dim,
                    window_size=cfg.window_size,
                    shift_window=cfg.shift_sequence[i % len(cfg.shift_sequence)],
                    qkv_bias=cfg.qkv_bias,
                    use_rope=cfg.use_rope,
                    rope_scale=cfg.rope_scale,
                    rope_theta=cfg.rope_theta,
                    qk_rms_norm=cfg.qk_rms_norm,
                    attn_backend=cfg.attn_backend,
                )
                for i in range(cfg.depth)
            ]
        )

        self.edge_head = EdgeFeatureHead(
            feat_channels=cfg.feat_dim,
            condition_dim=cfg.condition_dim,
            edge_dim=cfg.edge_feat_dim,
            hidden=cfg.edge_hidden_dim,
            mlp_depth=cfg.edge_mlp_depth,
            use_geom=cfg.edge_use_geom,
            pe_num_bands=cfg.edge_pe_num_bands,
            pe_max_freq=cfg.edge_pe_freq_max,
            use_cond=cfg.edge_use_condition,
        )

        self.coeff_head = LowRankMatrixCoefficientHead(
            feat_dim=cfg.latent_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            rank=cfg.lowrank_rank,
        )

        nbr_offsets = lattice_ball_offsets(float(cfg.neighbor_radius), device="cpu")
        self.nbr_offsets: torch.Tensor = Buffer(nbr_offsets, persistent=False)

    @torch.no_grad()
    def build_neighbors(
        self,
        coords_src: torch.Tensor,
        coords_dst: torch.Tensor,
        pos_dst_in_src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find src neighbors for each transformed destination row.

        Args:
            coords_src: [Nsrc, 4] source sparse coordinates.
            coords_dst: [Ndst, 4] destination sparse coordinates.
            pos_dst_in_src: [Ndst, 3] destination positions expressed in the
                source coordinate system.

        Returns:
            e_ids_dst: [E] edge -> destination row index.
            e_ids_src: [E] edge -> source row index.
        """
        device = coords_src.device
        nbr_offsets = self.nbr_offsets.to(device=device)

        query_bid = coords_dst[:, 0]
        key_bid = coords_src[:, 0]
        key_coords = coords_src[:, 1:]

        coord_min = key_coords.min(dim=0).values.to(dtype=torch.int32)
        e_ids_dst, e_ids_src = radius_nbr_edges(
            query_pos=pos_dst_in_src,
            query_bid=query_bid.to(dtype=torch.int32),
            key_coords=key_coords.to(dtype=torch.int32),
            key_bid=key_bid.to(dtype=torch.int32),
            radius=float(self.cfg.neighbor_radius),
            nbr_offsets=nbr_offsets.to(dtype=torch.int32),
            coord_min=coord_min,
        )
        return e_ids_dst, e_ids_src

    def forward(
        self,
        coords_src: torch.Tensor,
        coords_dst: torch.Tensor,
        O_dst2src: torch.Tensor,
        t_dst2src: torch.Tensor,
        s_dst2src: torch.Tensor,
    ) -> LinearCoefficient:
        """Predict the linear coefficient mapping source rows to destination rows.

        Args:
            coords_src: [Nsrc, 4] int32 source sparse coordinates.
            coords_dst: [Ndst, 4] int32 destination sparse coordinates.
            O_dst2src: [num_conditions, 3, 3] orthogonal matrices from
                destination coordinates to source coordinates.
            t_dst2src: [num_conditions, 3] translations in source grid-index
                units.
            s_dst2src: [num_conditions] integer orientation sign tokens.

        Returns:
            `LinearCoefficient` with `num_src == Nsrc` and `num_dst == Ndst`.
        """
        if coords_src.ndim != 2 or coords_src.shape[1] != 4:
            raise ValueError("coords_src must have shape [Nsrc, 4]")
        if coords_dst.ndim != 2 or coords_dst.shape[1] != 4:
            raise ValueError("coords_dst must have shape [Ndst, 4]")

        pos_src = coords_src[:, 1:].float()
        pos_dst = coords_dst[:, 1:].float()

        condition_rows_dst = coords_dst[:, 0].long()
        pos_dst_in_src = torch.bmm(O_dst2src[condition_rows_dst], pos_dst[..., None])[..., 0] + t_dst2src[condition_rows_dst]
        coords_dst_in_src = torch.cat(
            [
                coords_dst[:, :1],
                torch.floor(pos_dst_in_src).to(dtype=coords_dst.dtype),
            ],
            dim=1,
        )

        window_indices = build_swin_indices(
            coords=coords_dst_in_src,
            ctx_coords=coords_src,
            window_size=self.cfg.window_size,
            shift_sequence=list(self.cfg.shift_sequence),
        )
        condition = self.pose_conditioner(O_dst2src, t_dst2src, s_dst2src)

        feats_src = self.in_norm(self.in_proj(self.pos_pe(pos_src)))
        feats_dst = self.in_norm(self.in_proj(self.pos_pe(pos_dst_in_src)))

        for block in self.blocks:
            feats_src, feats_dst = block(
                feats_src=feats_src,
                pos_src=pos_src,
                coords_dst=coords_dst_in_src,
                feats_dst=feats_dst,
                pos_dst=pos_dst_in_src,
                condition=condition,
                window_indices=window_indices,
            )

        e_ids_dst, e_ids_src = self.build_neighbors(
            coords_src=coords_src,
            coords_dst=coords_dst,
            pos_dst_in_src=pos_dst_in_src,
        )
        edge_feat = self.edge_head(
            feats_src=feats_src,
            pos_src=pos_src,
            coords_dst=coords_dst_in_src,
            feats_dst=feats_dst,
            pos_dst=pos_dst_in_src,
            condition=condition,
            e_ids_dst=e_ids_dst,
            e_ids_src=e_ids_src,
        )

        return self.coeff_head(
            num_src=coords_src.shape[0],
            num_dst=coords_dst.shape[0],
            edge_feat=edge_feat,
            e_ids_dst=e_ids_dst,
            e_ids_src=e_ids_src,
        )


def build_swin_3d_latent_mapper(
    scale: Swin3DLatentMapperScale = "small",
    latent_dim: int = 8,
    lowrank_rank: Optional[int] = None,
) -> Swin3DLatentMapper:
    """Build a preset Swin 3D latent mapper."""

    cfg = swin_3d_latent_mapper_config(
        scale=scale,
        latent_dim=latent_dim,
        lowrank_rank=lowrank_rank,
    )
    return Swin3DLatentMapper(cfg)


__all__ = [
    "Swin3DLatentMapper",
    "build_swin_3d_latent_mapper",
]

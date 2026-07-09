from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Buffer

from ..geometry import lattice_ball_offsets, radius_nbr_edges
from .attention.window_index import build_swin_indices
from .blocks import NeighborGraphLatentMapperBlock, Swin3DLatentMapperBlock
from .config import (
    NeighborGraphLatentMapperConfig,
    NeighborGraphLatentMapperScale,
    Swin3DLatentMapperConfig,
    Swin3DLatentMapperScale,
    neighbor_graph_latent_mapper_config,
    swin_3d_latent_mapper_config,
)
from .encodings.pose_condition import PoseConditioner
from .encodings.position import FourierPE
from .heads import EdgeFeatureHead, LowRankMatrixCoefficientHead
from .operator import LinearCoefficient


class BaseSpatialTransformLatentMapper(nn.Module):
    """Interface for spatial-transform latent mappers."""

    def forward(
        self,
        coords_src: torch.Tensor,
        coords_dst: torch.Tensor,
        O_dst2src: torch.Tensor,
        t_dst2src: torch.Tensor,
        s_dst2src: torch.Tensor,
    ) -> LinearCoefficient:
        raise NotImplementedError


class Swin3DLatentMapper(BaseSpatialTransformLatentMapper):
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


def edges_to_csr(
    e_qids: torch.Tensor,
    e_kids: torch.Tensor,
    num_queries: int,
    num_keys: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_key = e_qids.to(dtype=torch.long) * int(num_keys) + e_kids.to(dtype=torch.long)
    order = torch.argsort(edge_key)
    q_sorted = e_qids[order].to(dtype=torch.long)
    col = e_kids[order].to(dtype=torch.int32).contiguous()
    counts = torch.bincount(q_sorted, minlength=num_queries)
    rowptr = torch.empty((num_queries + 1,), device=e_qids.device, dtype=torch.int32)
    rowptr[0] = 0
    rowptr[1:] = torch.cumsum(counts, dim=0).to(dtype=torch.int32)
    return rowptr, col


class NeighborGraphLatentMapper(BaseSpatialTransformLatentMapper):
    """Neighbor-graph implementation of the Spatial-Transform Latent Mapper.

    Predict a sparse src-to-dst latent mapping from a spatial transform.

    Coordinates are relation-expanded sparse lattice coordinates with shape
    [N, 4]. The first column indexes the transform/condition row; the remaining
    three columns are integer grid coordinates. The destination coordinates are
    transformed into the source coordinate system before graph attention and
    neighbor search.

    The returned `LinearCoefficient` maps source latent rows to destination
    latent rows in the input row order.
    """

    def __init__(self, cfg: NeighborGraphLatentMapperConfig) -> None:
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
                NeighborGraphLatentMapperBlock(
                    feat_dim=cfg.feat_dim,
                    num_heads=cfg.num_heads,
                    condition_dim=cfg.condition_dim,
                    qkv_bias=cfg.qkv_bias,
                    use_rope=cfg.use_rope,
                    rope_scale=cfg.rope_scale,
                    rope_theta=cfg.rope_theta,
                    qk_rms_norm=cfg.qk_rms_norm,
                )
                for _ in range(cfg.depth)
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

    def forward(
        self,
        coords_src: torch.Tensor,
        coords_dst: torch.Tensor,
        O_dst2src: torch.Tensor,
        t_dst2src: torch.Tensor,
        s_dst2src: torch.Tensor,
        sort: bool = True,
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
            sort: Whether to sort rows internally by `(relation_id, z, y, x)`
                for graph attention locality.

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

        condition = self.pose_conditioner(O_dst2src, t_dst2src, s_dst2src)

        feats_src = self.in_norm(self.in_proj(self.pos_pe(pos_src)))
        feats_dst = self.in_norm(self.in_proj(self.pos_pe(pos_dst_in_src)))

        if sort:
            src_order = torch.arange(coords_src.shape[0], device=coords_src.device, dtype=torch.long)
            src_order = src_order[torch.argsort(pos_src[src_order, 0], stable=True)]
            src_order = src_order[torch.argsort(pos_src[src_order, 1], stable=True)]
            src_order = src_order[torch.argsort(pos_src[src_order, 2], stable=True)]
            src_order = src_order[torch.argsort(coords_src[src_order, 0], stable=True)]

            dst_order = torch.arange(coords_dst.shape[0], device=coords_dst.device, dtype=torch.long)
            dst_order = dst_order[torch.argsort(pos_dst_in_src[dst_order, 0], stable=True)]
            dst_order = dst_order[torch.argsort(pos_dst_in_src[dst_order, 1], stable=True)]
            dst_order = dst_order[torch.argsort(pos_dst_in_src[dst_order, 2], stable=True)]
            dst_order = dst_order[torch.argsort(coords_dst[dst_order, 0], stable=True)]

            coords_src_attn = coords_src[src_order]
            pos_src_attn = pos_src[src_order]
            feats_src_attn = feats_src[src_order]
            coords_dst_attn = coords_dst_in_src[dst_order]
            pos_dst_attn = pos_dst_in_src[dst_order]
            coords_dst_graph = coords_dst[dst_order]
            pos_dst_graph = pos_dst[dst_order]
            feats_dst_attn = feats_dst[dst_order]
        else:
            coords_src_attn = coords_src
            pos_src_attn = pos_src
            feats_src_attn = feats_src
            coords_dst_attn = coords_dst_in_src
            pos_dst_attn = pos_dst_in_src
            coords_dst_graph = coords_dst
            pos_dst_graph = pos_dst
            feats_dst_attn = feats_dst

        device = coords_src.device
        nbr_offsets = self.nbr_offsets.to(device=device, dtype=torch.int32)
        neighbor_radius = float(self.cfg.neighbor_radius)

        src_key_coords = coords_src_attn[:, 1:].to(dtype=torch.int32)
        src_bid = coords_src_attn[:, 0].to(dtype=torch.int32)
        src_coord_min = src_key_coords.min(dim=0).values.to(dtype=torch.int32)
        e_src_qids, e_src_kids = radius_nbr_edges(
            query_pos=pos_src_attn,
            query_bid=src_bid,
            key_coords=src_key_coords,
            key_bid=src_bid,
            radius=neighbor_radius,
            nbr_offsets=nbr_offsets,
            coord_min=src_coord_min,
        )
        rowptr_src, col_src = edges_to_csr(
            e_src_qids,
            e_src_kids,
            coords_src_attn.shape[0],
            coords_src_attn.shape[0],
        )

        dst_key_coords = coords_dst_graph[:, 1:].to(dtype=torch.int32)
        dst_bid = coords_dst_graph[:, 0].to(dtype=torch.int32)
        dst_coord_min = dst_key_coords.min(dim=0).values.to(dtype=torch.int32)
        e_dst_qids, e_dst_kids = radius_nbr_edges(
            query_pos=pos_dst_graph,
            query_bid=dst_bid,
            key_coords=dst_key_coords,
            key_bid=dst_bid,
            radius=neighbor_radius,
            nbr_offsets=nbr_offsets,
            coord_min=dst_coord_min,
        )
        rowptr_dst, col_dst = edges_to_csr(
            e_dst_qids,
            e_dst_kids,
            coords_dst_attn.shape[0],
            coords_dst_attn.shape[0],
        )

        e_cross_qids, e_cross_kids = radius_nbr_edges(
            query_pos=pos_dst_attn,
            query_bid=dst_bid,
            key_coords=src_key_coords,
            key_bid=src_bid,
            radius=neighbor_radius,
            nbr_offsets=nbr_offsets,
            coord_min=src_coord_min,
        )
        rowptr_cross, col_cross = edges_to_csr(
            e_cross_qids,
            e_cross_kids,
            coords_dst_attn.shape[0],
            coords_src_attn.shape[0],
        )

        for block in self.blocks:
            feats_src_attn, feats_dst_attn = block(
                feats_src=feats_src_attn,
                pos_src=pos_src_attn,
                coords_dst=coords_dst_attn,
                feats_dst=feats_dst_attn,
                pos_dst=pos_dst_attn,
                condition=condition,
                rowptr_src=rowptr_src,
                col_src=col_src,
                rowptr_cross=rowptr_cross,
                col_cross=col_cross,
                rowptr_dst=rowptr_dst,
                col_dst=col_dst,
            )

        if sort:
            feats_src = torch.empty_like(feats_src_attn)
            feats_dst = torch.empty_like(feats_dst_attn)
            feats_src[src_order] = feats_src_attn
            feats_dst[dst_order] = feats_dst_attn
        else:
            feats_src = feats_src_attn
            feats_dst = feats_dst_attn

        query_bid = coords_dst[:, 0]
        key_bid = coords_src[:, 0]
        key_coords = coords_src[:, 1:]

        coord_min = key_coords.min(dim=0).values.to(dtype=torch.int32)
        e_ids_dst, e_ids_src = radius_nbr_edges(
            query_pos=pos_dst_in_src,
            query_bid=query_bid.to(dtype=torch.int32),
            key_coords=key_coords.to(dtype=torch.int32),
            key_bid=key_bid.to(dtype=torch.int32),
            radius=neighbor_radius,
            nbr_offsets=nbr_offsets,
            coord_min=coord_min,
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


def build_neighbor_graph_latent_mapper(
    scale: NeighborGraphLatentMapperScale = "small",
    latent_dim: int = 8,
    lowrank_rank: Optional[int] = None,
) -> NeighborGraphLatentMapper:
    """Build a preset neighbor-graph latent mapper."""

    cfg = neighbor_graph_latent_mapper_config(
        scale=scale,
        latent_dim=latent_dim,
        lowrank_rank=lowrank_rank,
    )
    return NeighborGraphLatentMapper(cfg)


__all__ = [
    "BaseSpatialTransformLatentMapper",
    "NeighborGraphLatentMapper",
    "Swin3DLatentMapper",
    "build_neighbor_graph_latent_mapper",
    "build_swin_3d_latent_mapper",
]

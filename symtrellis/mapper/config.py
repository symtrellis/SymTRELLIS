from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Swin3DLatentMapperScale = Literal["tiny", "small", "base"]


@dataclass(frozen=True)
class Swin3DLatentMapperConfig:
    """Configuration for the Swin 3D Spatial Transform Latent Mapper backend.

    The mapper trunk is grid-size agnostic: sparse coordinates may be arbitrary
    int32 lattice coordinates, including negative values.
    """

    # Transformer trunk.
    feat_dim: int = 128
    num_heads: int = 4
    depth: int = 6

    window_size: Tuple[int, int, int] = (4, 4, 4)
    shift_sequence: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 0),
        (2, 2, 2),
    )

    qkv_bias: bool = True
    use_rope: bool = True
    rope_scale: float = 0.75
    rope_theta: float = 10000.0
    qk_rms_norm: bool = True
    attn_backend: Literal["xformers", "flash_attn"] = "xformers"

    # Pose/transform conditioning.
    condition_dim: int = 128
    condition_hidden_dim: int = 256
    condition_num_layers: int = 2
    t_num_freqs: int = 10
    t_include_input: bool = False
    t_freq_min: float = 1.0
    sign_dim: int = 8

    # Initial sparse position embedding before the transformer trunk.
    pos_pe_num_bands: int = 6
    pos_pe_freq_max: float = 0.4
    pos_pe_include_input: bool = False

    # Edge feature head.
    edge_feat_dim: int = 64
    edge_hidden_dim: int = 128
    edge_mlp_depth: int = 2
    edge_use_geom: bool = True
    edge_pe_num_bands: int = 6
    edge_pe_freq_max: float = 0.5
    edge_use_condition: bool = True

    # Linear coefficient head.
    latent_dim: int = 8
    lowrank_rank: int = 8

    # Neighbor search for coefficient edges.
    neighbor_radius: float = 2.0


def swin_3d_latent_mapper_config(
    scale: Swin3DLatentMapperScale = "small",
    latent_dim: int = 8,
    lowrank_rank: Optional[int] = None,
) -> Swin3DLatentMapperConfig:
    """Return one preset Swin 3D latent mapper config by scale name."""

    if scale == "tiny":
        return Swin3DLatentMapperConfig(
            feat_dim=96,
            num_heads=4,
            depth=4,
            condition_dim=96,
            edge_feat_dim=48,
            latent_dim=latent_dim,
            lowrank_rank=4 if lowrank_rank is None else lowrank_rank,
        )

    if scale == "small":
        return Swin3DLatentMapperConfig(
            feat_dim=128,
            num_heads=4,
            depth=6,
            condition_dim=128,
            edge_feat_dim=64,
            latent_dim=latent_dim,
            lowrank_rank=8 if lowrank_rank is None else lowrank_rank,
        )

    if scale == "base":
        return Swin3DLatentMapperConfig(
            feat_dim=192,
            num_heads=6,
            depth=8,
            condition_dim=192,
            edge_feat_dim=96,
            latent_dim=latent_dim,
            lowrank_rank=8 if lowrank_rank is None else lowrank_rank,
        )

    raise ValueError(f"Unknown Swin3DLatentMapper scale: {scale}")


__all__ = [
    "Swin3DLatentMapperConfig",
    "Swin3DLatentMapperScale",
    "swin_3d_latent_mapper_config",
]

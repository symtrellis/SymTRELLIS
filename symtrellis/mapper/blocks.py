from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn

from .attention.csr_attn import CSRMultiHeadAttention
from .attention.window_attn import WindowMultiHeadAttention
from .attention.window_index import WindowIndex
from .layers import SparseAdaLayerNorm, SparseFFN


class Swin3DLatentMapperBlock(nn.Module):
    """Two-branch Swin-style 3D latent mapper block.

    The source branch is context/reference latent tokens. It is updated by
    source self-attention followed by an FFN. The destination branch is the
    latent tokens being mapped; it first cross-attends to the updated source
    branch, then applies destination self-attention and an FFN.

    Attention is windowed in sparse 3D grid space. This block does not build
    window partitions itself; the caller must pass `window_indices` produced by
    `build_swin_indices` for the same `shift_window`.

    Data flow:
        feats_src -> source self-attn -> source FFN
        feats_dst -> dst-to-src cross-attn -> dst self-attn -> dst FFN

    Normalization:
        Source branch uses plain LayerNorm. Destination branch uses
        SparseAdaLayerNorm, where `coords_dst[:, 0]` indexes rows in
        `condition`.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        condition_dim: int,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        shift_window: Tuple[int, int, int] = (0, 0, 0),
        qkv_bias: bool = True,
        use_rope: bool = True,
        rope_scale: float = 0.75,
        rope_theta: float = 10000.0,
        qk_rms_norm: bool = False,
        attn_backend: Literal["xformers", "flash_attn"] = "xformers",
    ) -> None:
        """
        Args:
            feat_dim: Feature width for both source and destination tokens.
            num_heads: Number of attention heads.
            condition_dim: Width of each destination conditioning row.
            window_size: 3D attention window size in grid cells.
            shift_window: 3D shifted-window offset used by this block.
            qkv_bias: Whether q/k/v projections use bias.
            use_rope: Whether to apply rotary position embedding to q/k.
            rope_scale: RoPE scale passed to `RotaryPositionEmbedder`.
            rope_theta: RoPE theta base passed to `RotaryPositionEmbedder`.
            qk_rms_norm: Whether to apply per-head RMS normalization to q/k.
            attn_backend: Window attention backend.
        """
        super().__init__()

        assert feat_dim % num_heads == 0

        self.feat_dim = feat_dim
        self.window_size = window_size
        self.condition_dim = condition_dim

        self.shift_window = shift_window

        # source branch
        self.norm1_src = nn.LayerNorm(feat_dim)
        self.self_attn_src = WindowMultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            ctx_feat_dim=None,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            qk_rms_norm=qk_rms_norm,
            attn_backend=attn_backend,
        )
        self.norm2_src = nn.LayerNorm(feat_dim)
        self.ffn_src = SparseFFN(feat_dim)
        self.norm_cross_src = nn.LayerNorm(feat_dim)

        # destination branch
        self.norm1_dst = SparseAdaLayerNorm(feat_dim, condition_dim)
        self.cross_attn_dst = WindowMultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            ctx_feat_dim=feat_dim,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            qk_rms_norm=qk_rms_norm,
            attn_backend=attn_backend,
        )
        self.norm2_dst = SparseAdaLayerNorm(feat_dim, condition_dim)
        self.self_attn_dst = WindowMultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            ctx_feat_dim=None,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            qk_rms_norm=qk_rms_norm,
            attn_backend=attn_backend,
        )
        self.norm3_dst = SparseAdaLayerNorm(feat_dim, condition_dim)
        self.ffn_dst = SparseFFN(feat_dim)

    def forward(
        self,
        feats_src: torch.Tensor,
        pos_src: torch.Tensor,
        coords_dst: torch.Tensor,
        feats_dst: torch.Tensor,
        pos_dst: torch.Tensor,
        condition: torch.Tensor,
        window_indices: Dict[str, WindowIndex],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one source/destination mapper block.

        Args:
            feats_src: Source feature tensor with shape [Nsrc, feat_dim].
            pos_src: Source positions with shape [Nsrc, 3], in the coordinate
                system expected by RoPE.
            coords_dst: Destination coordinates with shape [Ndst, 4].
                `coords_dst[:, 0]` indexes rows in `condition`; `coords_dst[:, 1:]`
                are integer grid coordinates used when building window indices.
            feats_dst: Destination feature tensor with shape [Ndst, feat_dim].
            pos_dst: Destination positions with shape [Ndst, 3].
            condition: Conditioning table with shape
                [num_conditions, condition_dim].
            window_indices: Dict returned by `build_swin_indices`. It must
                contain `self_src_{sx}_{sy}_{sz}`, `self_dst_{sx}_{sy}_{sz}`,
                and `cross_{sx}_{sy}_{sz}` for this block's `shift_window`.

        Returns:
            Updated `(feats_src, feats_dst)`, with shapes
            [Nsrc, feat_dim] and [Ndst, feat_dim].
        """

        shift_x, shift_y, shift_z = self.shift_window
        window_key_src = f"self_src_{shift_x}_{shift_y}_{shift_z}"
        window_key_dst = f"self_dst_{shift_x}_{shift_y}_{shift_z}"
        window_key_cross = f"cross_{shift_x}_{shift_y}_{shift_z}"

        required_window_keys = (window_key_src, window_key_dst, window_key_cross)
        assert all(n in window_indices for n in required_window_keys)

        # source branch
        feats_src = feats_src + self.self_attn_src(
            window_index=window_indices[window_key_src],
            x_feats=self.norm1_src(feats_src),
            x_pos=pos_src,
        )
        feats_src = feats_src + self.ffn_src(self.norm2_src(feats_src))

        # destination branch
        feats_dst = feats_dst + self.cross_attn_dst(
            window_index=window_indices[window_key_cross],
            x_feats=self.norm1_dst(coords_dst, feats_dst, condition),
            x_pos=pos_dst,
            ctx_feats=self.norm_cross_src(feats_src),
            ctx_pos=pos_src,
        )
        feats_dst = feats_dst + self.self_attn_dst(
            window_index=window_indices[window_key_dst],
            x_feats=self.norm2_dst(coords_dst, feats_dst, condition),
            x_pos=pos_dst,
        )
        feats_dst = feats_dst + self.ffn_dst(self.norm3_dst(coords_dst, feats_dst, condition))

        return feats_src, feats_dst


class NeighborGraphLatentMapperBlock(nn.Module):
    """Two-branch neighbor-graph latent mapper block.

    The source branch is context/reference latent tokens. It is updated by
    source self-attention followed by an FFN. The destination branch is the
    latent tokens being mapped; it first cross-attends to the updated source
    branch, then applies destination self-attention and an FFN.

    Attention is directed CSR graph attention. The caller owns graph
    construction and passes CSR tensors for source self-attention,
    destination-to-source cross-attention, and destination self-attention.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        condition_dim: int,
        qkv_bias: bool = True,
        use_rope: bool = True,
        rope_scale: float = 0.75,
        rope_theta: float = 10000.0,
        qk_rms_norm: bool = False,
    ) -> None:
        """
        Args:
            feat_dim: Feature width for both source and destination tokens.
            num_heads: Number of attention heads.
            condition_dim: Width of each destination conditioning row.
            qkv_bias: Whether q/k/v projections use bias.
            use_rope: Whether to apply rotary position embedding to q/k.
            rope_scale: RoPE scale passed to `RotaryPositionEmbedder`.
            rope_theta: RoPE theta base passed to `RotaryPositionEmbedder`.
            qk_rms_norm: Whether to apply per-head RMS normalization to q/k.
        """
        super().__init__()

        assert feat_dim % num_heads == 0

        self.feat_dim = feat_dim
        self.condition_dim = condition_dim

        # source branch
        self.norm1_src = nn.LayerNorm(feat_dim)
        self.self_attn_src = CSRMultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            ctx_feat_dim=None,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            qk_rms_norm=qk_rms_norm,
        )
        self.norm2_src = nn.LayerNorm(feat_dim)
        self.ffn_src = SparseFFN(feat_dim)
        self.norm_cross_src = nn.LayerNorm(feat_dim)

        # destination branch
        self.norm1_dst = SparseAdaLayerNorm(feat_dim, condition_dim)
        self.cross_attn_dst = CSRMultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            ctx_feat_dim=feat_dim,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            qk_rms_norm=qk_rms_norm,
        )
        self.norm2_dst = SparseAdaLayerNorm(feat_dim, condition_dim)
        self.self_attn_dst = CSRMultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            ctx_feat_dim=None,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            qk_rms_norm=qk_rms_norm,
        )
        self.norm3_dst = SparseAdaLayerNorm(feat_dim, condition_dim)
        self.ffn_dst = SparseFFN(feat_dim)

    def forward(
        self,
        feats_src: torch.Tensor,
        pos_src: torch.Tensor,
        coords_dst: torch.Tensor,
        feats_dst: torch.Tensor,
        pos_dst: torch.Tensor,
        condition: torch.Tensor,
        rowptr_src: torch.Tensor,
        col_src: torch.Tensor,
        rowptr_cross: torch.Tensor,
        col_cross: torch.Tensor,
        rowptr_dst: torch.Tensor,
        col_dst: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one source/destination neighbor-graph mapper block."""

        # source branch
        feats_src = feats_src + self.self_attn_src(
            rowptr=rowptr_src,
            col=col_src,
            x_feats=self.norm1_src(feats_src),
            x_pos=pos_src,
        )
        feats_src = feats_src + self.ffn_src(self.norm2_src(feats_src))

        # destination branch
        feats_dst = feats_dst + self.cross_attn_dst(
            rowptr=rowptr_cross,
            col=col_cross,
            x_feats=self.norm1_dst(coords_dst, feats_dst, condition),
            x_pos=pos_dst,
            ctx_feats=self.norm_cross_src(feats_src),
            ctx_pos=pos_src,
        )
        feats_dst = feats_dst + self.self_attn_dst(
            rowptr=rowptr_dst,
            col=col_dst,
            x_feats=self.norm2_dst(coords_dst, feats_dst, condition),
            x_pos=pos_dst,
        )
        feats_dst = feats_dst + self.ffn_dst(self.norm3_dst(coords_dst, feats_dst, condition))

        return feats_src, feats_dst

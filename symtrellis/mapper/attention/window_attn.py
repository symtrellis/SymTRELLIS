from typing import Literal, Optional, Tuple

import flash_attn
import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops.fmha import cutlass
from xformers.ops.fmha.attn_bias import BlockDiagonalMask, _SeqLenInfo

from ..encodings.position import RotaryPositionEmbedder
from ..layers import SparseMultiHeadRMSNorm
from .window_index import WindowIndex


def window_attn_backend_xformers(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu: torch.Tensor,
    k_cu: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Run xFormers block-diagonal window attention from GPU cu-seqlens.

    Args:
        q: Query tensor with shape [total_q, num_heads, head_dim]. Rows must be
            sorted into contiguous window blocks.
        k: Key tensor with shape [total_kv, num_heads, head_dim], sorted by the
            same window order as `q`.
        v: Value tensor with shape [total_kv, num_heads, head_dim], sorted like
            `k`.
        q_cu: Int32 tensor with shape [num_windows + 1]. Cumulative query row
            starts, where `q_cu[0] == 0` and `q_cu[-1] == total_q`.
        k_cu: Int32 tensor with shape [num_windows + 1]. Cumulative key/value
            row starts, where `k_cu[-1] == total_kv`.
        max_seqlen_q: Static upper bound for query tokens per window.
        max_seqlen_k: Static upper bound for key/value tokens per window.

    Returns:
        Tensor with shape [total_q, num_heads, head_dim].

    This intentionally avoids `BlockDiagonalMask.from_seqlens(...)`, because
    that path needs Python length lists and would require `tolist()` on CUDA
    tensors. `_SeqLenInfo` is a private xFormers API; `seqstart` carries the
    real GPU cu-seqlens, while `seqstart_py` is dummy Python metadata. Because
    some xFormers backends inspect `seqstart_py`, this path must force CUTLASS.
    """
    qq = q.unsqueeze(0).contiguous()
    kk = k.unsqueeze(0).contiguous()
    vv = v.unsqueeze(0).contiguous()

    # xFormers expects [B, M, H, D]. We use B=1 and let cu-seqlens define
    # block boundaries on GPU, without converting window lengths to Python lists.
    q_seqinfo = _SeqLenInfo(
        seqstart=q_cu,
        max_seqlen=max_seqlen_q,
        min_seqlen=1,
        seqstart_py=[0] * q_cu.shape[0],
    )
    k_seqinfo = _SeqLenInfo(
        seqstart=k_cu,
        max_seqlen=max_seqlen_k,
        min_seqlen=1,
        seqstart_py=[0] * k_cu.shape[0],
    )
    bias = BlockDiagonalMask(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    return xops.memory_efficient_attention(
        qq,
        kk,
        vv,
        attn_bias=bias,
        p=0.0,
        op=(cutlass.FwOp, cutlass.BwOp),
    )[0]


def window_attn_backend_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu: torch.Tensor,
    k_cu: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Run flash-attn varlen window attention from GPU cu-seqlens.

    Args:
        q: Query tensor with shape [total_q, num_heads, head_dim]. Rows must be
            sorted into contiguous window blocks.
        k: Key tensor with shape [total_kv, num_heads, head_dim], sorted by the
            same window order as `q`.
        v: Value tensor with shape [total_kv, num_heads, head_dim], sorted like
            `k`.
        q_cu: Int32 tensor with shape [num_windows + 1]. Cumulative query row
            starts, where `q_cu[0] == 0` and `q_cu[-1] == total_q`.
        k_cu: Int32 tensor with shape [num_windows + 1]. Cumulative key/value
            row starts, where `k_cu[-1] == total_kv`.
        max_seqlen_q: Static upper bound for query tokens per window.
        max_seqlen_k: Static upper bound for key/value tokens per window.

    Returns:
        Tensor with shape [total_q, num_heads, head_dim].

    flash-attn's varlen API accepts GPU cu-seqlens directly, so this backend
    does not need `tolist()` or `max().item()`. The max sequence lengths are
    passed as static Python integers from the configured window volume.
    """
    if q.dtype == torch.float32:
        raise ValueError("flash-attn usually does not support float32. Use xformers for fp32.")

    kv = torch.stack([k, v], dim=1).contiguous()  # [Mk,2,H,Dh]

    return flash_attn.flash_attn_varlen_kvpacked_func(  # type: ignore
        q,
        kv,
        cu_seqlens_q=q_cu,
        cu_seqlens_k=k_cu,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        causal=False,
    )


class WindowMultiHeadAttention(nn.Module):
    """Windowed sparse multi-head attention over grid tokens.

    This module consumes a precomputed `WindowIndex`; it does not build window
    partitions inside `forward`. Keeping index construction outside makes cache
    reuse explicit and avoids rebuilding GPU cu-seqlens inside every attention
    call.

    `x_*` tensors are the query/destination branch. If `ctx_*` tensors are not
    provided, the module runs self-attention and key/value rows also come from
    `x_*`. If `ctx_*` tensors are provided, the module runs cross-attention and
    key/value rows come from `ctx_*`.

    Args:
        feat_dim: Feature width of the query branch.
        num_heads: Number of attention heads. `feat_dim` must be divisible by
            `num_heads`.
        ctx_feat_dim: Feature width of the context branch. Defaults to
            `feat_dim` for self-attention.
        window_size: Window side lengths in grid cells. Used only to provide a
            static max sequence length to the backend.
        shift_window: Window shift used by the matching `WindowIndex`.
        qkv_bias: Whether to use bias in query/key/value projections.
        use_rope: Whether to apply rotary position embedding to q and k.
        rope_scale: Scale factor used by `RotaryPositionEmbedder`.
        rope_theta: Base theta used by `RotaryPositionEmbedder`.
        qk_rms_norm: Whether to apply per-head RMS normalization to q and k.
        attn_backend: Backend name. `xformers` supports fp32 training;
            `flash_attn` is intended for non-fp32 varlen attention.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        ctx_feat_dim: Optional[int] = None,
        window_size: Tuple[int, int, int] = (3, 3, 3),
        shift_window: Tuple[int, int, int] = (0, 0, 0),
        qkv_bias: bool = True,
        use_rope: bool = True,
        rope_scale: float = 0.75,
        rope_theta: float = 10000.0,
        qk_rms_norm: bool = False,
        attn_backend: Literal["xformers", "flash_attn"] = "xformers",
    ) -> None:
        super().__init__()

        assert feat_dim % num_heads == 0
        assert attn_backend in ["xformers", "flash_attn"]

        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_heads
        self.ctx_feat_dim = ctx_feat_dim if ctx_feat_dim is not None else feat_dim
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_window = shift_window
        self.max_window_tokens = window_size[0] * window_size[1] * window_size[2]
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.attn_backend = attn_backend

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
        window_index: WindowIndex,
        x_feats: torch.Tensor,  # torch.float32/16, [N, feat_dim]
        x_pos: torch.Tensor,  # torch.float32 [N, 3]
        ctx_feats: Optional[torch.Tensor] = None,
        ctx_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply window attention using a caller-provided `WindowIndex`.

        Args:
            window_index: Required `WindowIndex`. `q_rows` indexes `x_feats` and
                `x_pos`. In self-attention, `kv_rows` also indexes `x_feats` and
                `x_pos`; in cross-attention, `kv_rows` indexes `ctx_feats` and
                `ctx_pos`.
            x_feats: Query feature tensor with shape [Nq, feat_dim].
            x_pos: Query position tensor with shape [Nq, 3], in the coordinate
                system expected by the rotary position embedder.
            ctx_feats: Optional context feature tensor with shape
                [Nkv, ctx_feat_dim].
            ctx_pos: Optional context position tensor with shape [Nkv, 3].

        Returns:
            Tensor with shape [Nq, feat_dim]. Rows listed in
            `window_index.q_rows` receive attended values; rows absent from
            `q_rows` are zero before the final output projection.
        """

        assert (ctx_feats is None) == (ctx_pos is None)

        # Select the key/value branch. The window index, not this module,
        # defines which rows are grouped into each attention block.
        kv_feats = x_feats if ctx_feats is None else ctx_feats
        kv_pos = x_pos if ctx_pos is None else ctx_pos

        # Rows are already sorted into contiguous per-window blocks.
        q_rows = window_index.q_rows
        q_cu = window_index.q_cu
        kv_rows = window_index.kv_rows
        kv_cu = window_index.kv_cu

        # Project gathered rows, then reshape to varlen attention layout
        # [total_tokens, num_heads, head_dim].
        q = self.to_q(x_feats[q_rows])
        kv = self.to_kv(kv_feats[kv_rows])
        kv = kv.reshape(*kv.shape[:-1], 2, -1)
        k, v = kv.unbind(dim=-2)

        q = q.reshape(*q.shape[:-1], self.num_heads, -1)  # [Nf, Nh, head_dim]
        k = k.reshape(*k.shape[:-1], self.num_heads, -1)
        v = v.reshape(*v.shape[:-1], self.num_heads, -1)

        if self.use_rope:
            q = self.rope(q, x_pos[q_rows])
            k = self.rope(k, kv_pos[kv_rows])

        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # Backend calls consume GPU cu-seqlens directly; no Python seqlen lists
        # are constructed here.
        if self.attn_backend == "xformers":
            out = window_attn_backend_xformers(
                q,
                k,
                v,
                q_cu,
                kv_cu,
                self.max_window_tokens,
                self.max_window_tokens,
            )
        elif self.attn_backend == "flash_attn":
            out = window_attn_backend_flash_attn(
                q,
                k,
                v,
                q_cu,
                kv_cu,
                self.max_window_tokens,
                self.max_window_tokens,
            )
        else:
            raise ValueError(f"Unknown attention module: {self.attn_backend}")

        out = out.reshape(*out.shape[:-2], self.feat_dim)

        # Scatter attended query rows back to the original query row layout.
        base = out.new_zeros((x_feats.shape[0], out.shape[-1]))
        out = base.index_copy_(0, q_rows, out)

        return self.to_out(out)

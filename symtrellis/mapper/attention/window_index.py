"""Window partition indexing for sparse attention."""

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch


class WindowIndex(NamedTuple):
    """Row order and sequence metadata for windowed attention.

    The rows are sorted into contiguous per-window blocks. For window `i`,
    query rows are `q_rows[q_cu[i]:q_cu[i + 1]]`, and key/value rows are
    `kv_rows[kv_cu[i]:kv_cu[i + 1]]`.

    In self-attention, `q_rows` and `kv_rows` refer to the same input rows. In
    cross-attention, `q_rows` indexes `coords`, while `kv_rows` indexes
    `ctx_coords`; only windows present on both sides are kept. The cached
    min/max sequence lengths are Python integers for attention backends that
    need exact varlen metadata.
    """

    q_rows: torch.Tensor  # [Nq_kept], query rows sorted by window
    q_lens: torch.Tensor  # int32 [W], number of query rows per kept window
    q_cu: torch.Tensor  # int32 [W + 1], cumulative starts from q_lens
    q_min_seqlen: int  # minimum number of query rows in a kept window
    q_max_seqlen: int  # maximum number of query rows in a kept window
    kv_rows: torch.Tensor  # [Nkv_kept], key/value rows sorted by window
    kv_lens: torch.Tensor  # int32 [W], number of key/value rows per kept window
    kv_cu: torch.Tensor  # int32 [W + 1], cumulative starts from kv_lens
    kv_min_seqlen: int  # minimum number of key/value rows in a kept window
    kv_max_seqlen: int  # maximum number of key/value rows in a kept window


def build_window_index(
    coords: torch.Tensor,
    ctx_coords: Optional[torch.Tensor],
    window_size: Tuple[int, int, int],
    shift_window: Tuple[int, int, int],
) -> WindowIndex:
    """Build the row order and cu-seqlens for sparse window attention.

    For self-attention, query and key/value rows are identical, so one window
    partition is reused for both sides. For cross-attention, query-only and
    key/value-only windows are dropped because block attention needs matching
    q and kv sequences for each window.

    Args:
        coords: Integer tensor with shape [Nq, 4]. `coords[:, 0]` is the
            batch/sample id; `coords[:, 1:]` are integer grid coordinates.
            Negative grid coordinates are allowed.
        ctx_coords: Optional integer tensor with shape [Nkv, 4]. When given,
            these rows are used as key/value coordinates. It must use the same
            coordinate convention and device as `coords`.
        window_size: Three integer window side lengths in grid cells.
        shift_window: Three integer offsets added before floor-dividing by
            `window_size`.

    Returns:
        A `WindowIndex`. `q_rows` indexes `coords`; `kv_rows` indexes
        `ctx_coords` for cross-attention and `coords` for self-attention.
        `q_lens/q_cu`, `kv_lens/kv_cu`, and cached min/max sequence lengths are
        ready for varlen attention backends such as xformers or flash-attn.
    """
    device = coords.device

    win_size = torch.tensor(window_size, device=device, dtype=coords.dtype)[None]
    win_shift = torch.tensor(shift_window, device=device, dtype=coords.dtype)[None]

    # A window key is (batch/sample id, window_x, window_y, window_z).
    q_win = torch.div(coords[:, 1:] + win_shift, win_size, rounding_mode="floor")
    q_key = torch.cat([coords[:, :1], q_win], dim=1)

    if ctx_coords is None:
        # Self-attention: q and kv use the same rows and the same window ids.
        _, q_win_id, q_cnt = torch.unique(
            q_key,
            dim=0,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        q_rows = torch.argsort(q_win_id)
        q_lens = q_cnt.to(dtype=torch.int32)

        kv_rows = q_rows
        kv_lens = q_lens
    else:
        # Cross-attention: build one shared compact window id space for q and kv.
        kv_win = torch.div(
            ctx_coords[:, 1:] + win_shift,
            win_size,
            rounding_mode="floor",
        )
        kv_key = torch.cat([ctx_coords[:, :1], kv_win], dim=1)

        win_keys, win_id = torch.unique(
            torch.cat([q_key, kv_key], dim=0),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        num_q = q_key.shape[0]
        num_kv = kv_key.shape[0]
        num_win = win_keys.shape[0]

        q_win_id = win_id[:num_q]
        kv_win_id = win_id[num_q : num_q + num_kv]

        q_cnt = torch.bincount(q_win_id, minlength=num_win)
        kv_cnt = torch.bincount(kv_win_id, minlength=num_win)
        has_both = (q_cnt > 0) & (kv_cnt > 0)

        # Keep only windows that contain both query and key/value tokens.
        q_keep = has_both[q_win_id]
        q_keep_rows = torch.nonzero(q_keep).flatten()
        q_win_kept = q_win_id[q_keep]
        q_sort = torch.argsort(q_win_kept)
        q_rows = q_keep_rows[q_sort]
        q_lens = q_cnt[has_both].to(dtype=torch.int32)

        kv_keep = has_both[kv_win_id]
        kv_keep_rows = torch.nonzero(kv_keep).flatten()
        kv_win_kept = kv_win_id[kv_keep]
        kv_sort = torch.argsort(kv_win_kept)
        kv_rows = kv_keep_rows[kv_sort]
        kv_lens = kv_cnt[has_both].to(dtype=torch.int32)

    # Convert per-window lengths to cumulative sequence starts for varlen attention.
    q_cu = torch.empty((q_lens.numel() + 1,), device=device, dtype=torch.int32)
    q_cu[0] = 0
    q_cu[1:] = torch.cumsum(q_lens, dim=0)

    if ctx_coords is None:
        kv_cu = q_cu
    else:
        kv_cu = torch.empty((kv_lens.numel() + 1,), device=device, dtype=torch.int32)
        kv_cu[0] = 0
        kv_cu[1:] = torch.cumsum(kv_lens, dim=0)

    q_min, q_max = torch.aminmax(q_lens)
    q_min_seqlen = int(q_min.item())
    q_max_seqlen = int(q_max.item())
    if ctx_coords is None:
        kv_min_seqlen = q_min_seqlen
        kv_max_seqlen = q_max_seqlen
    else:
        kv_min, kv_max = torch.aminmax(kv_lens)
        kv_min_seqlen = int(kv_min.item())
        kv_max_seqlen = int(kv_max.item())

    return WindowIndex(
        q_rows=q_rows,
        q_lens=q_lens,
        q_cu=q_cu,
        q_min_seqlen=q_min_seqlen,
        q_max_seqlen=q_max_seqlen,
        kv_rows=kv_rows,
        kv_lens=kv_lens,
        kv_cu=kv_cu,
        kv_min_seqlen=kv_min_seqlen,
        kv_max_seqlen=kv_max_seqlen,
    )


def build_swin_indices(
    coords: torch.Tensor,
    ctx_coords: torch.Tensor,
    window_size: Tuple[int, int, int],
    shift_sequence: List[Tuple[int, int, int]],
) -> Dict[str, WindowIndex]:
    """Build the window indices used by a two-branch Swin-style block.

    Args:
        coords: Integer tensor with shape [Ndst, 4] for the destination/query
            branch.
        ctx_coords: Integer tensor with shape [Nsrc, 4] for the source/context
            branch.
        window_size: Three integer window side lengths in grid cells.
        shift_sequence: Window shifts to precompute. Each shift is a tuple
            `(sx, sy, sz)`.

    Returns:
        A dict mapping index names to `WindowIndex` objects. For each shift
        name `{sx}_{sy}_{sz}`, the keys are:
        `cross_{name}` for query=`coords`, kv=`ctx_coords`;
        `self_dst_{name}` for self-attention over `coords`;
        `self_src_{name}` for self-attention over `ctx_coords`.
        These keys are the contract consumed by `Swin3DLatentMapperBlock`.
    """
    results: Dict[str, WindowIndex] = {}
    for shift_window in shift_sequence:
        s1, s2, s3 = shift_window
        shift_name = f"{s1}_{s2}_{s3}"

        # Cross-attention from destination/query rows to source/context rows.
        results["cross_" + shift_name] = build_window_index(
            coords=coords,
            ctx_coords=ctx_coords,
            window_size=window_size,
            shift_window=shift_window,
        )

        # Self-attention inside the destination/query branch.
        results["self_dst_" + shift_name] = build_window_index(
            coords=coords,
            ctx_coords=None,
            window_size=window_size,
            shift_window=shift_window,
        )

        # Self-attention inside the source/context branch.
        results["self_src_" + shift_name] = build_window_index(
            coords=ctx_coords,
            ctx_coords=None,
            window_size=window_size,
            shift_window=shift_window,
        )

    return results

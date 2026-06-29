from typing import Tuple

import torch

def sparse_csr_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Directed CSR attention forward.

    Args:
        q: [Nq, H, D] projected query tensor.
        k: [Nk, H, D] projected key tensor.
        v: [Nk, H, D] projected value tensor.
        rowptr: [Nq + 1] int32 CSR row pointer.
        col: [E] int32 key/value row indices.

    Returns:
        Tuple `(out, lse)` where `out` is [Nq, H, D] and `lse` is [Nq, H]
        float32 log-sum-exp saved for backward.
    """
    ...

def sparse_csr_attn_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directed CSR attention backward.

    Args:
        grad_out: [Nq, H, D] gradient of forward output.
        q: [Nq, H, D] saved query tensor.
        k: [Nk, H, D] saved key tensor.
        v: [Nk, H, D] saved value tensor.
        rowptr: [Nq + 1] int32 CSR row pointer.
        col: [E] int32 key/value row indices.
        out: [Nq, H, D] saved forward output.
        lse: [Nq, H] saved forward log-sum-exp.

    Returns:
        Tuple `(dq, dk, dv)`. dK/dV accumulation uses CUDA atomics and is not
        bitwise deterministic.
    """
    ...

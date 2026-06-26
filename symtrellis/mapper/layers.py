import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMultiHeadRMSNorm(nn.Module):
    """
    Per-head normalization for sparse attention q/k tensors.

    Expected input shape is [N, heads, dim], where N is the flattened sparse
    token axis. Normalization is applied over the last dimension for each
    token/head pair, followed by a learned per-head scale.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
          feat: [N, heads, dim] sparse attention features.

        Returns:
          Normalized features with the same shape and dtype.
        """
        return (F.normalize(feat.float(), dim=-1) * self.gamma * self.scale).to(feat.dtype)


class SparseAdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm for sparse feature rows.

    coords[:, 0] is used as the row index into cond. In other words, the first
    coordinate column must already match the conditioning table.

    Shapes:
      feats: [N, dim].
      coords: [N, 4], integer sparse coordinates.
      cond: [B, cond_dim].
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-5,
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim
        self.eps = eps

        self.to_gamma_beta = nn.Linear(cond_dim, 2 * dim, bias=True)

        if zero_init:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LayerNorm per sparse row, then modulate by its conditioning row.

        Returns:
          [N, dim] features with the same dtype as feats.
        """
        assert feats.ndim == 2 and feats.shape[1] == self.dim
        assert coords.ndim == 2 and coords.shape[1] == 4
        assert cond.ndim == 2 and cond.shape[1] == self.cond_dim

        idx_batch = coords[:, 0]
        x = feats.float()

        # Normalize each sparse feature row independently.
        x = F.layer_norm(x, (self.dim,), weight=None, bias=None, eps=self.eps)

        # Use coords[:, 0] to select one affine modulation per sparse row.
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)  # [B, D], [B, D]
        y = x * (1.0 + gamma[idx_batch]) + beta[idx_batch]

        return y.to(dtype=feats.dtype)


class SparseFFN(nn.Module):
    """
    SwiGLU feed-forward network for sparse feature rows.

    Expected input and output shape is [N, dim].
    """

    def __init__(self, dim: int, hidden_mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = dim * hidden_mult

        self.fc1 = nn.Linear(self.dim, 2 * self.hidden, bias=bias)
        self.fc2 = nn.Linear(self.hidden, self.dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(u) * v
        return self.fc2(x)

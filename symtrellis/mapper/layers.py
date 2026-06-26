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

    coords[:, 0] is used as the row index into condition. In other words, the
    first coordinate column must already match the conditioning table.

    Shapes:
      feats: [N, dim].
      coords: [N, 4], integer sparse coordinates.
      condition: [num_conditions, condition_dim].
    """

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        eps: float = 1e-5,
        zero_init: bool = True,
    ) -> None:
        """
        Args:
            dim: Feature width.
            condition_dim: Width of each conditioning row.
            eps: LayerNorm epsilon.
            zero_init: If True, initialize the affine modulation to zero so the
                module starts as plain LayerNorm.
        """
        super().__init__()

        self.dim = dim
        self.condition_dim = condition_dim
        self.eps = eps

        self.to_gamma_beta = nn.Linear(condition_dim, 2 * dim, bias=True)

        if zero_init:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LayerNorm per sparse row, then modulate by its conditioning row.

        Args:
          coords: [N, 4] integer coordinates. `coords[:, 0]` indexes rows in
            `condition`.
          feats: [N, dim] sparse feature rows.
          condition: [num_conditions, condition_dim] conditioning table.

        Returns:
          [N, dim] features with the same dtype as feats.
        """
        assert feats.ndim == 2 and feats.shape[1] == self.dim
        assert coords.ndim == 2 and coords.shape[1] == 4
        assert condition.ndim == 2 and condition.shape[1] == self.condition_dim

        condition_rows = coords[:, 0]
        x = feats.float()

        # Normalize each sparse feature row independently.
        x = F.layer_norm(x, (self.dim,), weight=None, bias=None, eps=self.eps)

        # Use coords[:, 0] to select one affine modulation per sparse row.
        gamma, beta = self.to_gamma_beta(condition).chunk(2, dim=1)
        y = x * (1.0 + gamma[condition_rows]) + beta[condition_rows]

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

import math

import torch
from torch import nn
from torch.nn.parameter import Buffer


class PoseConditioner(nn.Module):
    """
    Build a pose condition vector from:
      1. the first two columns of R as a 6D continuous orientation feature,
      2. a periodic Fourier encoding of t,
      3. a learned embedding of the discrete orientation sign.

    Inputs:
      R: [B, 3, 3]
         Orthogonal transform matrix. May belong to O(3), not necessarily SO(3).

      t: [B, 3]
         Translation-like continuous variable already expressed in the chosen
         grid-unit coordinate system.
         If periodicity is desired, t must already be mapped into the intended
         period-1 domain before calling this module.

      s: [B]
         Discrete sign token in {0, 1}.
         Typically used to distinguish the two connected components of O(3),
         e.g. det(R) < 0 and det(R) > 0.

    Fourier encoding:
      Frequencies are dyadic harmonics:
          freq_k = freq_min * 2^k
      Frequencies are measured in cycles per unit. With freq_min = 1.0,
      the first harmonic has period 1.
      Periodicity is therefore guaranteed only if the input t itself is already
      represented modulo that period.

    Output:
      condition: [B, condition_dim]
    """

    def __init__(
        self,
        condition_dim: int,
        t_num_freqs: int = 10,
        t_include_input: bool = False,
        freq_min: float = 1.0,
        sign_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """
        Args:
            condition_dim: Output condition feature width.
            t_num_freqs: Number of dyadic Fourier frequencies for translation.
            t_include_input: Whether to append raw `t` to its Fourier features.
                If False, integer-period translations are intentionally aliased
                by the periodic encoding.
            freq_min: Lowest translation frequency in cycles per unit.
            sign_dim: Embedding width for the discrete orientation sign token.
            hidden_dim: Hidden width of the MLP.
            num_layers: Number of linear layers in the MLP.
        """

        super().__init__()
        self.condition_dim = condition_dim
        self.t_num_freqs = t_num_freqs
        self.t_include_input = t_include_input
        self.freq_min = float(freq_min)

        freqs = (2.0 ** torch.arange(self.t_num_freqs, dtype=torch.float32)) * self.freq_min
        self.t_freqs = Buffer(freqs, persistent=False)  # [F]

        self.sign_embed = nn.Embedding(2, sign_dim)

        t_dim = (3 if self.t_include_input else 0) + 3 * (2 * self.t_num_freqs)
        in_dim = 6 + t_dim + sign_dim

        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        layers.append(nn.Linear(d, self.condition_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
          R: [B, 3, 3]
          t: [B, 3], already represented in the intended periodic/grid-unit domain
          s: [B], integer token in {0, 1}

        Returns:
          condition: [B, condition_dim]
        """
        assert R.ndim == 3 and R.shape[-2:] == (3, 3)
        assert t.ndim == 2 and t.shape[1] == 3
        assert s.ndim == 1

        r6 = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)  # [B, 6]

        tf = (2.0 * math.pi) * t[:, :, None] * self.t_freqs[None, None, :]  # [B, 3, F]
        t_feat = torch.cat([tf.sin(), tf.cos()], dim=1).reshape(t.shape[0], -1)  # [B, 6F]

        if self.t_include_input:
            t_feat = torch.cat([t, t_feat], dim=1)

        s_feat = self.sign_embed(s)

        x = torch.cat([r6, t_feat, s_feat], dim=1)
        return self.mlp(x)  # [B, condition_dim]

"""Time-conditioned MLP shared by the diffusion and flow-matching models.

Both generative models use this exact backbone so the comparison isolates the training
objective and sampler, not the architecture. The network maps (x, t) -> R^dim, where
t is a float in [0, 1]; each method decides what t means and what the output predicts
(noise for DDPM, velocity for flow matching).
"""

import math

import torch
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Transformer-style embedding of t in [0, 1] (scaled by 1000 so low frequencies matter)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    angles = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.time = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        z = self.act(self.fc1(self.norm(h)) + self.time(temb))
        return h + self.fc2(z)


class TimeMLP(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 256, n_blocks: int = 4, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t, self.t_dim))
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, temb)
        return self.out(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

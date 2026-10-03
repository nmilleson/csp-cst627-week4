"""1D Fourier Neural Operator (Li et al., 2021) for the Burgers map u(x, 0) -> u(x, T).

Architecture:
  input [u0(x), x]  -> pointwise lift to `width` channels
  -> n_layers x  [ SpectralConv1d (lowest `modes` Fourier modes) + pointwise linear ] -> GELU
  -> pointwise projection to 1 channel

SpectralConv1d does rfft along x, multiplies the first `modes` coefficients by learned
complex weights, and irfft's back to the *input's own length*. The parameters live in
frequency space rather than on grid points, so the same weights can be applied on any
grid with at least 2 * modes points. Whether that resolution invariance actually holds
is tested later in the failure log.

Training uses the relative L2 loss from the FNO paper; `train_operator` logs train/test
error, gradient norm, wall-clock and peak VRAM.
"""

import math
import time

import numpy as np
import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes,
                                                       dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N)
        n = x.shape[-1]
        x_hat = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_hat.shape[-1])
        out_hat = torch.zeros(x.shape[0], self.weight.shape[1], x_hat.shape[-1],
                              dtype=torch.cfloat, device=x.device)
        out_hat[..., :modes] = torch.einsum("bik,iok->bok", x_hat[..., :modes],
                                            self.weight[..., :modes])
        return torch.fft.irfft(out_hat, n=n, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        self.modes, self.width = modes, width
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(n_layers))
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def grid(n: int, device) -> torch.Tensor:
        """Physical coordinates x_j = j / n on the periodic unit interval."""
        return torch.arange(n, device=device, dtype=torch.float32) / n

    def forward(self, u0: torch.Tensor) -> torch.Tensor:  # u0: (B, N) -> (B, N)
        x = self.grid(u0.shape[-1], u0.device).expand_as(u0)
        h = self.lift(torch.stack([u0, x], dim=-1)).permute(0, 2, 1)  # (B, W, N)
        for i, (spec, pw) in enumerate(zip(self.spectral, self.pointwise)):
            h = spec(h) + pw(h)
            if i < len(self.spectral) - 1:
                h = nn.functional.gelu(h)
        return self.project(h.permute(0, 2, 1)).squeeze(-1)


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-sample ||pred - true|| / ||true||, averaged over the batch."""
    return (torch.linalg.norm(pred - true, dim=-1) / torch.linalg.norm(true, dim=-1)).mean()


@torch.no_grad()
def predict(model, u0: np.ndarray, device: str = "cpu", batch: int = 200) -> np.ndarray:
    """Batched inference, NumPy (B, N) in and out."""
    model.eval()
    out = [model(torch.as_tensor(u0[s:s + batch], device=device)).cpu().numpy()
           for s in range(0, len(u0), batch)]
    return np.concatenate(out)


def per_sample_rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=-1) / np.linalg.norm(true, axis=-1)


def evaluate(model, u0: np.ndarray, uT: np.ndarray, device: str = "cpu") -> float:
    """Mean relative L2 error over a dataset (NumPy in, float out)."""
    return float(per_sample_rel_l2(predict(model, u0, device), uT).mean())


def train_operator(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    epochs: int = 300,
    batch_size: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float | None = None,
    eval_every: int = 10,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train `model` in place on (u0, uT) pairs with relative-L2 loss. Returns history."""
    torch.manual_seed(seed)
    model = model.to(device)
    u0 = torch.as_tensor(train[0], device=device)
    uT = torch.as_tensor(train[1], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(u0) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    hist = {"epoch": [], "train_rel_l2": [], "test_rel_l2": [], "grad_norm": [], "lr": [],
            "diverged_at": None}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(u0), device=device)
        ep_loss, ep_gn = 0.0, 0.0
        for s in range(0, len(u0), batch_size):
            idx = perm[s:s + batch_size]
            loss = relative_l2(model(u0[idx]), uT[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf"))
            opt.step()
            sched.step()
            ep_loss += loss.item() / steps_per_epoch
            ep_gn += gn.item() / steps_per_epoch
        if not math.isfinite(ep_loss):
            hist["diverged_at"] = epoch
            break
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            hist["epoch"].append(epoch + 1)
            hist["train_rel_l2"].append(ep_loss)
            hist["test_rel_l2"].append(evaluate(model, *test, device=device))
            hist["grad_norm"].append(ep_gn)
            hist["lr"].append(sched.get_last_lr()[0])

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        hist["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    else:
        hist["peak_vram_mb"] = None
    hist["train_seconds"] = time.perf_counter() - t0
    hist["config"] = dict(epochs=epochs, batch_size=batch_size, lr=lr,
                          weight_decay=weight_decay, grad_clip=grad_clip, seed=seed)
    return hist

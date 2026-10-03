"""Sample-quality metrics for the 2D generative models, plus a latency/VRAM profiler.

Quality is measured against the held-out checkerboard reference set:
  * support_fraction - share of samples on a black square (1.0 = perfect)
  * sliced Wasserstein-2 distance (SWD) - full distributional mismatch
  * MMD^2 with a multi-bandwidth RBF kernel - sensitive to missing/extra mass

SWD and MMD are never exactly 0 for finite samples, so `evaluate_samples` also reports
a "floor": the same metric between two independent draws from the true distribution.
A model is as good as the data allows when it reaches the floor.
"""

import time

import numpy as np
import torch

from halden.data2d import sample_checkerboard, support_fraction


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 256, seed: int = 0) -> float:
    """Sliced Wasserstein-2 distance between equal-size point clouds (truncates to the smaller)."""
    n = min(len(x), len(y))
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((x.shape[1], n_proj))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    px = np.sort(x[:n] @ dirs, axis=0)
    py = np.sort(y[:n] @ dirs, axis=0)
    return float(np.sqrt(np.mean((px - py) ** 2)))


def mmd_rbf(
    x: np.ndarray,
    y: np.ndarray,
    bandwidths: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0),
    n_max: int = 5000,
) -> float:
    """Unbiased MMD^2 with a sum of RBF kernels. Subsamples to n_max points per set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.as_tensor(x[:n_max], dtype=torch.float64, device=device)
    y = torch.as_tensor(y[:n_max], dtype=torch.float64, device=device)
    dxx, dyy, dxy = torch.cdist(x, x) ** 2, torch.cdist(y, y) ** 2, torch.cdist(x, y) ** 2
    m, n = len(x), len(y)
    total = 0.0
    for h in bandwidths:
        kxx, kyy, kxy = (torch.exp(-d / (2 * h**2)) for d in (dxx, dyy, dxy))
        total += ((kxx.sum() - m) / (m * (m - 1))
                  + (kyy.sum() - n) / (n * (n - 1))
                  - 2 * kxy.mean()).item()
    return total


def evaluate_samples(samples: np.ndarray, ref: np.ndarray) -> dict:
    """All quality metrics for one sample set against the reference set."""
    samples = np.asarray(samples)
    finite = np.isfinite(samples).all(axis=1)
    s = samples[finite]
    return {
        "support_fraction": float(support_fraction(s) * finite.mean()),
        "swd": sliced_wasserstein(s, ref),
        "mmd": mmd_rbf(s, ref),
        "nonfinite_fraction": float(1.0 - finite.mean()),
    }


def metric_floor(ref: np.ndarray, seed: int = 123) -> dict:
    """Metrics for a fresh independent draw of the true distribution: the best achievable."""
    fresh = sample_checkerboard(len(ref), np.random.default_rng(seed))
    return evaluate_samples(fresh, ref)


def measure(fn, device: str, warmup_fn=None, repeats: int = 1) -> tuple[object, float, float | None]:
    """Run fn() `repeats` times. Returns (last result, mean seconds, peak VRAM MiB or None).

    `warmup_fn` (e.g. fn on a tiny batch) runs first so CUDA init/kernel selection is not
    counted in the latency.
    """
    is_cuda = device.startswith("cuda")
    if warmup_fn is not None:
        warmup_fn()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if is_cuda else None
    return result, float(np.mean(times)), peak

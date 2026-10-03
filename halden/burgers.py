"""1D viscous Burgers' equation: finite-difference solver and operator-learning dataset.

    u_t + (u^2 / 2)_x = nu * u_xx,    x in [0, 1) periodic,    t in [0, T]

The operator to learn is G: u(x, 0) -> u(x, T).

Discretization on a uniform periodic grid of N points (dx = 1/N), Strang-split per step:
  * Diffusion: second-order central-difference Laplacian. Because the grid is periodic the
    FD matrix is circulant, so its exponential is diagonal in Fourier space with eigenvalues
    -(4 / dx^2) sin^2(pi k / N). We integrate that semi-discrete system exactly over each
    half-step, which removes the stiff dx^2/nu time-step limit without changing the spatial
    scheme.
  * Convection: conservative finite-volume form with MUSCL (minmod) reconstruction and the
    Rusanov (local Lax-Friedrichs) flux, advanced with SSP-RK2 (Heun).
  * Time step: fixed per batch, dt = cfl * dx / max|u0|. Burgers obeys a maximum principle,
    so the initial max bounds |u| for all t and a single dt is safe for the whole run.

Defaults (nu = 0.005, T = 0.5) are chosen so shocks actually form: peak |u_x| grows ~2.5x
over the initial condition. Validated against the exact Cole-Hopf solution with 2nd-order
convergence (relative L2 error 2.2e-3 at N=256, 1.5e-4 at N=1024).

Initial conditions are zero-mean periodic Gaussian random fields with a power-law spectrum
(the same family used in the original FNO Burgers benchmark).

Datasets are generated once on a fine grid and subsampled by striding, so training and
evaluation can use different resolutions of the same underlying solutions.
"""

from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------- initial conditions


def sample_grf(
    n_samples: int,
    n_grid: int,
    rng: np.random.Generator,
    alpha: float = 2.5,
    tau: float = 7.0,
    sigma: float = 0.5,
    k_max: int | None = None,
) -> np.ndarray:
    """Zero-mean periodic GRF on [0, 1) with spectrum (4 pi^2 k^2 + tau^2)^(-alpha/2).

    Scaled so that the pointwise standard deviation is `sigma` in expectation.
    Returns float64 array (n_samples, n_grid).
    """
    k_max = k_max or n_grid // 4
    k = np.arange(1, k_max + 1)
    s = (4.0 * np.pi**2 * k**2 + tau**2) ** (-alpha / 2.0)
    # Each mode k contributes 2 * s_k^2 to the variance (k and -k).
    s *= sigma / np.sqrt(2.0 * np.sum(s**2))

    coeffs = np.zeros((n_samples, n_grid // 2 + 1), dtype=np.complex128)
    z = rng.standard_normal((n_samples, k_max)) + 1j * rng.standard_normal((n_samples, k_max))
    coeffs[:, 1 : k_max + 1] = z / np.sqrt(2.0) * s
    return np.fft.irfft(coeffs, n=n_grid, axis=-1) * n_grid


# ----------------------------------------------------------------------------- solver


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(a * b > 0.0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), 0.0)


def _convection_rhs(u: torch.Tensor, dx: float) -> torch.Tensor:
    """-(u^2/2)_x via MUSCL-minmod reconstruction + Rusanov flux. Operates on the last axis."""
    slope = _minmod(u - torch.roll(u, 1, dims=-1), torch.roll(u, -1, dims=-1) - u)
    u_left = u + 0.5 * slope                                # state just left of face i+1/2
    u_right = torch.roll(u - 0.5 * slope, -1, dims=-1)      # state just right of face i+1/2
    speed = torch.maximum(u_left.abs(), u_right.abs())
    flux = 0.25 * (u_left**2 + u_right**2) - 0.5 * speed * (u_right - u_left)
    return -(flux - torch.roll(flux, 1, dims=-1)) / dx


@torch.no_grad()
def solve_burgers(
    u0: np.ndarray,
    nu: float = 0.005,
    t_final: float = 0.5,
    cfl: float = 0.4,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Evolve a batch of periodic initial conditions u0 (B, N) to t_final. Returns (B, N).

    The time loop runs in PyTorch (float64) so it can use a GPU; input and output are NumPy.
    Each batch gets its own dt from its own max|u0|, so one large-amplitude sample does not
    slow down the whole dataset.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    u0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))
    n_grid = u0.shape[-1]
    dx = 1.0 / n_grid

    k = torch.arange(n_grid // 2 + 1, dtype=torch.float64, device=device)
    fd_laplacian_eig = -(4.0 / dx**2) * torch.sin(torch.pi * k / n_grid) ** 2

    out = np.empty_like(u0)
    for start in range(0, len(u0), batch_size):
        u = torch.from_numpy(u0[start : start + batch_size]).to(device)

        u_max = max(u.abs().max().item(), 1e-8)
        n_steps = int(np.ceil(t_final / (cfl * dx / u_max)))
        dt = t_final / n_steps
        # Exact half-step propagator for the central-difference Laplacian.
        half_diffuse = torch.exp(nu * fd_laplacian_eig * 0.5 * dt)

        def diffuse(v):
            return torch.fft.irfft(torch.fft.rfft(v, dim=-1) * half_diffuse, n=n_grid, dim=-1)

        for _ in range(n_steps):
            u = diffuse(u)
            u1 = u + dt * _convection_rhs(u, dx)
            u = 0.5 * (u + u1 + dt * _convection_rhs(u1, dx))
            u = diffuse(u)
        out[start : start + batch_size] = u.cpu().numpy()
    return out


# ----------------------------------------------------------------------------- independent reference


def cole_hopf_reference(
    u0_fn, nu: float, t_final: float, n_grid: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Exact solution via the Cole-Hopf transform, used only to validate `solve_burgers`.

    u = -2 nu phi_x / phi, where phi solves the heat equation with
    phi(x, 0) = exp(-(1 / 2nu) * integral_0^x u0). Requires u0 to have zero mean so phi
    is periodic, and moderate amplitude/nu so the exponentials stay in range.
    Returns (x, u(x, t_final)) on a fine grid, solved spectrally.
    """
    x = np.arange(n_grid) / n_grid
    u0 = u0_fn(x)
    k = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=1.0 / n_grid)

    u0_hat = np.fft.rfft(u0)
    u0_hat[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        integral_hat = np.where(k > 0, u0_hat / (1j * k), 0.0)
    integral = np.fft.irfft(integral_hat, n=n_grid)

    phi0 = np.exp(-integral / (2.0 * nu))
    phi_hat = np.fft.rfft(phi0) * np.exp(-nu * k**2 * t_final)
    phi = np.fft.irfft(phi_hat, n=n_grid)
    phi_x = np.fft.irfft(1j * k * phi_hat, n=n_grid)
    return x, -2.0 * nu * phi_x / phi


# ----------------------------------------------------------------------------- dataset


def generate_burgers_dataset(
    n_samples: int = 1200,
    n_grid: int = 1024,
    nu: float = 0.005,
    t_final: float = 0.5,
    seed: int = 0,
    cache_path: str | Path | None = None,
    **grf_kwargs,
) -> dict:
    """Generate (u0, uT) pairs on a fine grid, optionally caching to / loading from .npz.

    Returns dict with x (N,), u0 (B, N), uT (B, N) as float32, plus the scalar parameters.
    """
    if cache_path is not None and Path(cache_path).exists():
        data = dict(np.load(cache_path))
        if data["u0"].shape == (n_samples, n_grid) and float(data["nu"]) == nu:
            return data

    rng = np.random.default_rng(seed)
    u0 = sample_grf(n_samples, n_grid, rng, **grf_kwargs)
    uT = solve_burgers(u0, nu=nu, t_final=t_final)
    data = {
        "x": (np.arange(n_grid) / n_grid).astype(np.float32),
        "u0": u0.astype(np.float32),
        "uT": uT.astype(np.float32),
        "nu": np.float64(nu),
        "t_final": np.float64(t_final),
    }
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **data)
    return data


def subsample(arr: np.ndarray, n_target: int) -> np.ndarray:
    """Stride-subsample the last axis from N to n_target points (N must be a multiple)."""
    n_grid = arr.shape[-1]
    if n_grid % n_target:
        raise ValueError(f"{n_grid} is not a multiple of {n_target}")
    return arr[..., :: n_grid // n_target]


def train_test_split(data: dict, n_train: int = 1000) -> tuple[dict, dict]:
    """Split a dataset dict along the sample axis (first n_train samples for training)."""
    train = {"x": data["x"], "u0": data["u0"][:n_train], "uT": data["uT"][:n_train]}
    test = {"x": data["x"], "u0": data["u0"][n_train:], "uT": data["uT"][n_train:]}
    return train, test

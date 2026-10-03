"""Rectified flow / flow matching (Liu et al., 2022; Lipman et al., 2022) on 2D points.

Linear interpolation between noise z ~ N(0, I) at t = 0 and data x at t = 1:

    x_t = (1 - t) z + t x,        target velocity  v = x - z

The network v_theta(x_t, t) regresses v; sampling integrates dx/dt = v_theta(x, t) from
t = 0 to 1 with a fixed-step ODE solver. With independently drawn (z, x) pairs this is
"1-rectified flow" (= conditional flow matching with straight paths).

Reflow: integrate the trained 1-RF from fresh noise to get deterministic couplings
(z, x_hat = ODE(z)), then train a new model on those fixed pairs. Paths of the coupled
pairs cross less, so the learned ODE is straighter and few-step sampling improves
("2-rectified flow"). `straightness` quantifies this.

Training reuses `halden.training.train_generative`: the loss closures take a batch of
rows from the data array. For reflow the "data" rows are [z, x_hat] concatenated.
"""

import torch


def make_rf_loss():
    """1-RF loss: independent noise per batch, t ~ U[0, 1]."""

    def loss_fn(model, x1):
        z = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1.0 - t[:, None]) * z + t[:, None] * x1
        return torch.mean((model(xt, t) - (x1 - z)) ** 2)

    return loss_fn


def make_reflow_loss(dim: int = 2):
    """Reflow loss: each data row is a fixed coupling [z, x_hat] of width 2 * dim."""

    def loss_fn(model, pairs):
        z, x1 = pairs[:, :dim], pairs[:, dim:]
        t = torch.rand(len(pairs), device=pairs.device)
        xt = (1.0 - t[:, None]) * z + t[:, None] * x1
        return torch.mean((model(xt, t) - (x1 - z)) ** 2)

    return loss_fn


def _step(model, x, t0: float, t1: float, method: str) -> torch.Tensor:
    h = t1 - t0
    v0 = model(x, torch.full((len(x),), t0, device=x.device))
    if method == "euler":
        return x + h * v0
    if method == "heun":
        x_pred = x + h * v0
        v1 = model(x_pred, torch.full((len(x),), t1, device=x.device))
        return x + 0.5 * h * (v0 + v1)
    raise ValueError(f"unknown method {method!r}")


def nfe_for(n_steps: int, method: str) -> int:
    """Network evaluations used by `rf_sample` (Heun costs two per step)."""
    return n_steps * (2 if method == "heun" else 1)


@torch.no_grad()
def rf_sample(
    model,
    n: int,
    n_steps: int = 50,
    method: str = "euler",
    dim: int = 2,
    device: str = "cpu",
    seed: int | None = None,
    z: torch.Tensor | None = None,
    record: bool = False,
) -> tuple[torch.Tensor, list]:
    """Integrate dx/dt = v(x, t) from noise (t=0) to data (t=1) on a uniform grid.

    Pass `z` to start from given noise (used for reflow couplings). Returns
    (samples, trajectory) where trajectory is the list of states (CPU) if record=True.
    """
    if z is None:
        gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
        z = torch.randn(n, dim, device=device, generator=gen)
    x = z.to(device)
    ts = torch.linspace(0.0, 1.0, n_steps + 1).tolist()
    traj = [x.cpu()] if record else []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        x = _step(model, x, t0, t1, method)
        if record:
            traj.append(x.cpu())
    return x, traj


@torch.no_grad()
def make_reflow_pairs(
    model, n: int, n_steps: int = 100, method: str = "heun", dim: int = 2,
    device: str = "cpu", seed: int = 1, batch: int = 50_000,
) -> torch.Tensor:
    """Couplings [z, ODE(z)] from a trained flow, as an (n, 2*dim) float32 tensor on CPU."""
    gen = torch.Generator(device).manual_seed(seed)
    out = []
    for start in range(0, n, batch):
        z = torch.randn(min(batch, n - start), dim, device=device, generator=gen)
        x, _ = rf_sample(model, len(z), n_steps, method, dim, device, z=z)
        out.append(torch.cat([z, x], dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def straightness(
    model, n: int = 5000, n_steps: int = 100, dim: int = 2, device: str = "cpu", seed: int = 0,
) -> float:
    """S = E_t E_z || (x_1 - z) - v(x_t, t) ||^2 along the model's own Euler trajectories.

    0 means every trajectory is a straight line traversed at constant speed, so a single
    Euler step would be exact. Larger values mean more curvature.
    """
    gen = torch.Generator(device).manual_seed(seed)
    z = torch.randn(n, dim, device=device, generator=gen)
    ts = torch.linspace(0.0, 1.0, n_steps + 1).tolist()
    x, vs = z, []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        v = model(x, torch.full((n,), t0, device=device))
        vs.append(v)
        x = x + (t1 - t0) * v
    chord = x - z
    return float(torch.stack([((chord - v) ** 2).sum(-1).mean() for v in vs]).mean())

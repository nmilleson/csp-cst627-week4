"""DDPM (Ho et al., 2020) on 2D points: noise schedules, epsilon-prediction loss, samplers.

Forward process:  x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   t = 0 .. T-1
Training target:  the network predicts eps from (x_t, t / (T-1))
Samplers:
  * `ddpm_sample` - ancestral sampling, all T steps (T network evaluations)
  * `ddim_sample` - DDIM (Song et al., 2021) on a strided subset of timesteps, so the same
    trained model can be sampled with far fewer network evaluations (NFE). eta = 0 is
    deterministic; eta = 1 recovers DDPM-like stochasticity.

The schedule is passed explicitly to both the loss and the samplers. Normally the same
object goes to both; passing a different one at sampling time is how the
"mismatched noise schedule" failure is induced later.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NoiseSchedule:
    name: str
    betas: torch.Tensor  # (T,) float64

    def __post_init__(self):
        self.betas = self.betas.double()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @property
    def T(self) -> int:
        return len(self.betas)

    def to(self, device) -> "NoiseSchedule":
        return NoiseSchedule(self.name, self.betas.to(device))

    def log_snr(self) -> torch.Tensor:
        return torch.log(self.alpha_bars / (1.0 - self.alpha_bars))


def make_schedule(kind: str = "linear", T: int = 1000, **kw) -> NoiseSchedule:
    """'linear': betas from beta_start to beta_end (Ho et al.).
    'cosine': abar(t) = cos^2(((t/T + s)/(1 + s)) * pi/2) (Nichol & Dhariwal), betas clipped.
    """
    if kind == "linear":
        betas = torch.linspace(kw.get("beta_start", 1e-4), kw.get("beta_end", 0.02), T,
                               dtype=torch.float64)
    elif kind == "cosine":
        s = kw.get("s", 0.008)
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        abar = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        betas = (1.0 - abar[1:] / abar[:-1]).clamp(max=kw.get("max_beta", 0.999))
    else:
        raise ValueError(f"unknown schedule {kind!r}")
    name = kw.get("name", kind)
    return NoiseSchedule(name, betas)


def fingerprint(sched: NoiseSchedule) -> str:
    """Short identifier of a schedule's exact betas, stored alongside model checkpoints."""
    digest = hashlib.sha1(sched.betas.cpu().numpy().tobytes()).hexdigest()[:10]
    return f"{sched.name}-T{sched.T}-{digest}"


def check_schedule(sched: NoiseSchedule, expected: str | None = None,
                   max_terminal_abar: float = 1e-3) -> None:
    """Guard to run before sampling. Raises ValueError if the schedule differs from the
    one the model was trained with, or if abar_T is too large for x_T ~ N(0, I) to hold."""
    if expected is not None and fingerprint(sched) != expected:
        raise ValueError(f"sampling schedule {fingerprint(sched)} != training schedule {expected}")
    abar_T = sched.alpha_bars[-1].item()
    if abar_T > max_terminal_abar:
        raise ValueError(f"abar_T = {abar_T:.2e} > {max_terminal_abar:.0e}: x_T is not pure noise")


def _t_input(t: torch.Tensor, T: int) -> torch.Tensor:
    """Integer timestep -> network time input in [0, 1]."""
    return t.float() / (T - 1)


def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: NoiseSchedule, eps=None) -> torch.Tensor:
    """Draw x_t ~ q(x_t | x_0) for integer timesteps t (shape (B,))."""
    eps = torch.randn_like(x0) if eps is None else eps
    abar = sched.alpha_bars.to(x0.device)[t].float()[:, None]
    return abar.sqrt() * x0 + (1.0 - abar).sqrt() * eps


def make_ddpm_loss(sched: NoiseSchedule):
    """Return loss_fn(model, x0) for `train_generative`: MSE between true and predicted eps."""
    cache = {}

    def loss_fn(model, x0):
        if x0.device not in cache:
            cache[x0.device] = sched.alpha_bars.to(x0.device).float()
        abar_all = cache[x0.device]
        t = torch.randint(sched.T, (len(x0),), device=x0.device)
        eps = torch.randn_like(x0)
        abar = abar_all[t][:, None]
        xt = abar.sqrt() * x0 + (1.0 - abar).sqrt() * eps
        return torch.mean((model(xt, _t_input(t, sched.T)) - eps) ** 2)

    return loss_fn


@torch.no_grad()
def ddpm_sample(
    model,
    sched: NoiseSchedule,
    n: int,
    dim: int = 2,
    device: str = "cpu",
    seed: int | None = None,
    record_every: int | None = None,
    clip_x0: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Ancestral sampling with posterior variance beta_tilde. Returns (x_0, trajectory).

    Each step forms x0_hat = (x_t - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t), optionally
    clips it to [-clip_x0, clip_x0], and takes the posterior mean of q(x_{t-1} | x_t, x0_hat).
    Without clipping this is exactly the standard eps-parameterized DDPM update.
    trajectory maps timestep -> samples (CPU) every `record_every` steps (empty if None).
    """
    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
    s = sched.to(device)
    x = torch.randn(n, dim, device=device, generator=gen)
    traj = {s.T: x.cpu()} if record_every else {}
    for t in reversed(range(s.T)):
        tt = torch.full((n,), t, device=device)
        eps = model(x, _t_input(tt, s.T)).double()
        beta, alpha, abar = s.betas[t], s.alphas[t], s.alpha_bars[t]
        abar_prev = s.alpha_bars[t - 1] if t > 0 else torch.ones_like(abar)
        xd = x.double()
        x0_hat = (xd - (1.0 - abar).sqrt() * eps) / abar.sqrt()
        if clip_x0 is not None:
            x0_hat = x0_hat.clamp(-clip_x0, clip_x0)
        mean = (abar_prev.sqrt() * beta / (1.0 - abar) * x0_hat
                + alpha.sqrt() * (1.0 - abar_prev) / (1.0 - abar) * xd)
        if t > 0:
            var = beta * (1.0 - abar_prev) / (1.0 - abar)
            noise = torch.randn(n, dim, device=device, generator=gen, dtype=torch.float64)
            x = (mean + var.sqrt() * noise).float()
        else:
            x = mean.float()
        if record_every and t % record_every == 0:
            traj[t] = x.cpu()  # x is now x_t
    return x, traj


@torch.no_grad()
def ddim_sample(
    model,
    sched: NoiseSchedule,
    n: int,
    n_steps: int = 50,
    eta: float = 0.0,
    dim: int = 2,
    device: str = "cpu",
    seed: int | None = None,
    clip_x0: float | None = None,
) -> torch.Tensor:
    """DDIM sampling over `n_steps` evenly spaced timesteps (NFE = n_steps).

    With clip_x0, x0_hat is clipped and eps is re-derived from it so the update stays
    consistent with the clipped prediction.
    """
    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
    s = sched.to(device)
    ts = np.unique(np.linspace(0, s.T - 1, n_steps).round().astype(int))[::-1]
    x = torch.randn(n, dim, device=device, generator=gen, dtype=torch.float64)
    for i, t in enumerate(ts):
        abar = s.alpha_bars[t]
        abar_prev = s.alpha_bars[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=device,
                                                                               dtype=torch.float64)
        tt = torch.full((n,), int(t), device=device)
        eps = model(x.float(), _t_input(tt, s.T)).double()
        x0_pred = (x - (1.0 - abar).sqrt() * eps) / abar.sqrt()
        if clip_x0 is not None:
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)
            eps = (x - abar.sqrt() * x0_pred) / (1.0 - abar).sqrt()
        sigma = eta * ((1.0 - abar_prev) / (1.0 - abar) * (1.0 - abar / abar_prev)).sqrt()
        x = (abar_prev.sqrt() * x0_pred
             + (1.0 - abar_prev - sigma**2).clamp(min=0.0).sqrt() * eps)
        if eta > 0 and i + 1 < len(ts):
            x = x + sigma * torch.randn(n, dim, device=device, generator=gen, dtype=torch.float64)
    return x.float()

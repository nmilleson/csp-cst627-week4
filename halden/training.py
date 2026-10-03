"""Shared training loop for the 2D generative models (DDPM and flow matching).

Each method supplies only a `loss_fn(model, x0_batch) -> scalar`. Everything else -
optimizer, LR schedule, batch sampling, EMA, logging, timing, VRAM accounting - is
identical, which is what makes the comparison apples-to-apples.

The history records loss and the pre-clip gradient norm at every log step, and the loop
stops early on a non-finite loss. Those signals are the evidence for the failure log.
"""

import copy
import math
import time

import numpy as np
import torch
from torch import nn


def cosine_lr(step: int, total: int, warmup: int) -> float:
    """Multiplier on the base LR: linear warmup then cosine decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1.0 - decay)


def train_generative(
    model: nn.Module,
    loss_fn,
    data: np.ndarray,
    *,
    steps: int = 20_000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    warmup: int = 500,
    grad_clip: float | None = None,
    ema_decay: float = 0.999,
    log_every: int = 100,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Train `model` in place and return (ema_model, history)."""
    torch.manual_seed(seed)
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    data_t = torch.as_tensor(data, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, steps, warmup))

    hist = {"step": [], "loss": [], "grad_norm": [], "lr": [], "diverged_at": None}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    running_loss, running_gn = 0.0, 0.0
    for step in range(steps):
        idx = torch.randint(len(data_t), (batch_size,), device=device)
        loss = loss_fn(model, data_t[idx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ with max_norm=inf just measures the norm without clipping.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf"))
        opt.step()
        sched.step()
        _ema_update(ema, model, ema_decay)

        running_loss += loss.item()
        running_gn += grad_norm.item()
        if not math.isfinite(loss.item()):
            hist["diverged_at"] = step
            hist["step"].append(step)
            hist["loss"].append(float("nan"))
            hist["grad_norm"].append(grad_norm.item())
            hist["lr"].append(sched.get_last_lr()[0])
            break
        if (step + 1) % log_every == 0:
            hist["step"].append(step + 1)
            hist["loss"].append(running_loss / log_every)
            hist["grad_norm"].append(running_gn / log_every)
            hist["lr"].append(sched.get_last_lr()[0])
            running_loss, running_gn = 0.0, 0.0

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        hist["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    else:
        hist["peak_vram_mb"] = None
    hist["train_seconds"] = time.perf_counter() - t0
    hist["config"] = dict(steps=steps, batch_size=batch_size, lr=lr, warmup=warmup,
                          grad_clip=grad_clip, ema_decay=ema_decay, seed=seed)
    return ema, hist

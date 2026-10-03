"""Tools for the deliberate-failure study: buggy variants, diagnostics, and the failure log.

Each failure is induced on purpose, diagnosed from recorded evidence, repaired, and given
a guard check that would catch it automatically before scaling up:

  F1  DDPM schedule mismatch     -> ddpm.check_schedule (fingerprint + terminal-SNR guard)
  F2  exploding learning rate    -> prediction_collapse (output spread vs target spread)
  F3  FNO resolution mismatch    -> resolution_consistency (predict fine, subsample, compare)
"""

import json
from pathlib import Path

import numpy as np
import torch

from halden.ddpm import NoiseSchedule, q_sample, _t_input
from halden.fno import FNO1d, predict


# ----------------------------------------------------------------------------- F1 diagnostics


@torch.no_grad()
def per_timestep_eps_error(model, x0: np.ndarray, sched: NoiseSchedule,
                           timesteps=range(0, 1000, 50), device: str = "cpu") -> np.ndarray:
    """Model's eps-prediction MSE when data is noised with `sched` at each timestep.

    With the training schedule this reproduces the training loss per t. With a different
    schedule, the network is shown a noise level it does not associate with that t, and
    the error rises wherever the two schedules disagree.
    """
    x0 = torch.as_tensor(x0, device=device)
    errs = []
    for t in timesteps:
        tt = torch.full((len(x0),), t, device=device)
        eps = torch.randn_like(x0)
        xt = q_sample(x0, tt, sched.to(device), eps)
        errs.append(torch.mean((model(xt, _t_input(tt, sched.T)) - eps) ** 2).item())
    return np.array(errs)


# ----------------------------------------------------------------------------- F2 diagnostics


@torch.no_grad()
def prediction_collapse(model, x1: np.ndarray, seed: int = 0, device: str = "cpu") -> dict:
    """Spread of the velocity model's output vs. the spread of its regression target.

    A collapsed network outputs (nearly) the same vector for every input: its loss sits at
    the "predict the mean" level and pred_std / target_std -> 0.
    """
    gen = torch.Generator().manual_seed(seed)
    x1 = torch.as_tensor(x1)
    z = torch.randn(x1.shape, generator=gen)
    t = torch.rand(len(x1), generator=gen)
    xt = (1 - t[:, None]) * z + t[:, None] * x1
    v = model(xt.to(device), t.to(device)).cpu()
    target = x1 - z
    pred_std, target_std = v.std(0).mean().item(), target.std(0).mean().item()
    return {"pred_std": pred_std, "target_std": target_std, "ratio": pred_std / target_std,
            "mean_baseline_loss": float(target.var(0).mean())}


def param_norm(model) -> float:
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())))


# ----------------------------------------------------------------------------- F3 buggy variant + guard


class IndexGridFNO(FNO1d):
    """FNO with a common bug: the coordinate channel is the grid *index* j, not x = j / N.

    At the training resolution this is just a rescaled coordinate and the model trains
    fine. At any other resolution the coordinate channel takes values never seen in
    training (e.g. up to 1023 instead of 127), so the "resolution-invariant" operator
    silently breaks.
    """

    @staticmethod
    def grid(n: int, device) -> torch.Tensor:
        return torch.arange(n, device=device, dtype=torch.float32)


def resolution_consistency(model, u0_fine: np.ndarray, coarse: int, device: str = "cpu") -> float:
    """Relative L2 gap between predict-at-fine-then-subsample and predict-at-coarse.

    No ground truth needed. A resolution-invariant operator gives ~0; a large value means
    the model's output depends on the grid it is evaluated on.
    """
    stride = u0_fine.shape[-1] // coarse
    fine_then_sub = predict(model, u0_fine, device)[:, ::stride]
    coarse_pred = predict(model, u0_fine[:, ::stride], device)
    gap = np.linalg.norm(fine_then_sub - coarse_pred, axis=-1) / np.linalg.norm(coarse_pred, axis=-1)
    return float(gap.mean())


# ----------------------------------------------------------------------------- the log


class FailureLog:
    """Structured record of each induced failure. Serializes to JSON and Markdown."""

    FIELDS = ("id", "model", "title", "induced_by", "symptom", "evidence_broken",
              "diagnosis", "root_cause", "repair", "evidence_repaired", "guard")

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, **entry) -> None:
        missing = [f for f in self.FIELDS if f not in entry]
        if missing:
            raise ValueError(f"failure log entry missing {missing}")
        self.entries = [e for e in self.entries if e["id"] != entry["id"]] + [entry]

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    def _evidence(self, ev: dict) -> str:
        return "\n".join(f"  - {k}: {self._fmt(v)}" for k, v in ev.items())

    def to_markdown(self) -> str:
        lines = ["# Failure log", "",
                 "| ID | Model | Failure | Symptom | Repair |", "|---|---|---|---|---|"]
        for e in self.entries:
            lines.append(f"| {e['id']} | {e['model']} | {e['title']} | {e['symptom']} | {e['repair']} |")
        for e in self.entries:
            lines += ["", f"## {e['id']}: {e['title']} ({e['model']})", "",
                      f"**Induced by:** {e['induced_by']}", "",
                      f"**Symptom:** {e['symptom']}", "",
                      "**Evidence (broken):**", self._evidence(e["evidence_broken"]), "",
                      f"**Diagnosis:** {e['diagnosis']}", "",
                      f"**Root cause:** {e['root_cause']}", "",
                      f"**Repair:** {e['repair']}", "",
                      "**Evidence (repaired):**", self._evidence(e["evidence_repaired"]), "",
                      f"**Guard before scaling:** {e['guard']}"]
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path = "results") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "failure_log.json").write_text(json.dumps(self.entries, indent=2, default=float))
        (directory / "failure_log.md").write_text(self.to_markdown(), encoding="utf-8")

# Decision memo: generative and operator-learning pilot

**To:** Kwame Boateng · **Hardware measured on:** Tesla T4 · **Mode:** full budget

## Recommendation

**Pilot the Fourier Neural Operator.** On the Burgers benchmark its error against the fine solver is 0.28% relative L2, versus 0.84% for the cheap classical solver, while running **73x faster** than that solver and **709x faster** than the fine solver. It also passed the resolution-consistency guard, so one trained model serves every grid tested.

**Of the generative recipes, 2-RF is the one to carry forward.** It reaches the quality target with 1 network evaluations, 0.8 ms per 1k samples (others: DDPM: 47.9 ms at NFE 50; 1-RF: 9.0 ms at NFE 10). DDPM also depends on clipping to known data bounds, which real wave-load fields won't provide.

The operator is the stronger pilot candidate for Halden because it directly replaces a cost we already pay, solver runs, and we measured that saving against the solver itself. The generative models were measured only against each other on a 2D toy distribution. They tell us which recipe to prefer, not yet whether either one earns money.

## Measured basis

### Operator: FNO vs. the solver it would replace
| Method | Time for test set (s) | ms / case | Speedup vs fine solver | rel. L2 vs fine solver | Peak VRAM (MiB) |
|---|---|---|---|---|---|
| FNO @ N=128 | 0.0048 | 0.024 | 709x | 0.0028 | 83 |
| FNO @ N=1024 | 0.0279 | 0.140 | 121x | 0.0028 | 303 |
| solver @ N=128 | 0.3455 | 1.727 | 10x | 0.0084 | 54 |
| solver @ N=1024 | 3.3714 | 16.857 | 1x | 0.0000 | 70 |

FNO trained at N=128 in 114s; at N=1024: mean rel. L2 0.0028, 95th pct 0.0057, worst 0.0472; spread of mean error across N=32..1024: 0.0031.

### Generative: DDPM vs. rectified flow (same data, network, training budget)
Quality target: SWD <= 0.033 (2x the metric floor 0.016).

| Recipe | Train wall-clock (s) | Train peak VRAM (MiB) | Best SWD (support) @ NFE | NFE to hit target | Latency at target (ms / 1k samples) | Needs known data bounds |
|---|---|---|---|---|---|---|
| DDPM | 131 | 79 | 0.015 (0.984) @ 1000 | 50 | 47.9 | yes (x0 clipping) |
| 1-RF | 127 | 90 | 0.012 (0.970) @ 50 | 10 | 9.0 | no |
| 2-RF | 274 | 101 | 0.012 (0.972) @ 25 | 1 | 0.8 | no |

2-RF wall-clock includes training the 1-RF it distills from and generating its coupling pairs.

## Failure study

Three failures were induced, diagnosed and repaired. Each now has an automated guard (full log: `results/failure_log.md`):
- **F1 (DDPM): Mismatched noise schedule (train linear, sample cosine).** Guard: checkpoint stores schedule_fingerprint (sha1 of betas); ddpm.check_schedule raises on mismatch and on abar_T > 1e-3. Report SWD/MMD, not support fraction alone.
- **F2 (Rectified flow (1-RF)): Exploding learning rate.** Guard: LR range test before each new scale/architecture; abort a run whose loss stays within 5% of the closed-form constant-predictor loss; log parameter norm (87x baseline here) as a second alarm. Don't count on clipping with Adam.
- **F3 (FNO-1d): Resolution mismatch between training and evaluation grids.** Guard: resolution_consistency(model, u0_fine, coarse) < 1e-2 at every deployment grid, checked in CI; always report error at more than one grid.

## What must be checked before scaling


**Applies to every recipe**
1. **Seed variance.** Every number here is from a single seed. Repeat with at least 3 seeds and report mean ± spread. A difference smaller than the seed-to-seed spread doesn't count as a difference.
2. **Timing on the target hardware.** Colab GPUs are shared and timings are noisy. Re-measure latency, wall-clock and peak VRAM on the pilot GPU type, with repeats.
3. **Keep the guards on.** LR range test plus the constant-predictor abort (F2), schedule fingerprint plus terminal-SNR check (F1), and resolution-consistency gap (F3) run automatically on every new configuration.

**Operator pilot (FNO)**
4. **Real boundary conditions and geometry.** The FFT assumes a periodic domain; platform wave loading is not periodic. Verify accuracy with domain padding or a geometry-aware variant before trusting the speedup.
5. **Out-of-distribution inputs.** Training inputs came from one random-field family at one viscosity. Test storm-like extremes and shifted parameters, and agree an acceptance threshold on the engineering quantity (for example peak load), not on mean L2; compare the worst-case error in the scorecard with that threshold.
6. **Break-even on training data.** The surrogate needed 1000 solver runs to train. Compute queries-to-break-even = (data-generation + training cost) / (per-query saving) at pilot fidelity; the pilot only pays off above that query volume.
7. **Scaling with dimension.** Spectral weights grow with modes^d in 2D/3D. Measure parameters, VRAM and latency at the real grid size before committing cluster time.

**Generative recipe (if pursued)**
8. **Domain metrics.** SWD and MMD do not scale to high-dimensional fields, and F1 showed that coarse metrics can pass a broken model. Define physics-based checks: spectra, exceedance probabilities and extreme-value statistics.
9. **No hidden data assumptions.** DDPM's quality here relies on clipping to known data bounds; confirm the chosen recipe works without them on real fields.
10. **Reflow ceiling.** 2-RF can only be as good as the 1-RF that generated its pairs; confirm that the speed gain is worth any quality loss at pilot scale.


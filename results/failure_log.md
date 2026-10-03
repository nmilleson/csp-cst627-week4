# Failure log

| ID | Model | Failure | Symptom | Repair |
|---|---|---|---|---|
| F1 | DDPM | Mismatched noise schedule (train linear, sample cosine) | No error raised. Support 0.718 -> 0.641, SWD 0.091 -> 0.329 (3.6x); sample std 0.85 vs reference 1.15; centre-cell mass 0.157, corner 0.031 (true 0.125). | Sample with the exact schedule used in training (the Part 1 linear schedule). |
| F2 | Rectified flow (1-RF) | Exploding learning rate | Loss and grad-norm spike in the first logging window, then a calm-looking plateau at the constant-predictor loss; no NaN, so nothing crashes. Samples collapse. | lr = 1e-3 with 500-step linear warmup and cosine decay; range test: LRs that avoid collapse [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03], lowest final loss at 1e-02. |
| F3 | FNO-1d | Resolution mismatch between training and evaluation grids | Plausible error on the training grid (0.027) but 0.616 at N=1024 and 0.254 at N=64; no error raised. | Use physical coordinates x_j = j/N on the periodic unit interval (FNO1d.grid). |

## F1: Mismatched noise schedule (train linear, sample cosine) (DDPM)

**Induced by:** Sampled the linear-schedule DDPM checkpoint with make_schedule('cosine').

**Symptom:** No error raised. Support 0.718 -> 0.641, SWD 0.091 -> 0.329 (3.6x); sample std 0.85 vs reference 1.15; centre-cell mass 0.157, corner 0.031 (true 0.125).

**Evidence (broken):**
  - support_fraction: 0.6411
  - swd: 0.3295
  - mmd: 0.09789
  - sample_std: 0.8464
  - reference_std: 1.153
  - corner_cell_mass: 0.0313
  - centre_cell_mass: 0.1573
  - ddim50_swd: 0.2243
  - eps_mse_mid_t_cosine: 0.6483
  - eps_mse_mid_t_linear: 0.1008

**Diagnosis:** Support fraction moves far less than SWD/MMD and per-cell mass, so it is not a sufficient check. Per-timestep eps-error under the sampling schedule is far above the training-schedule error at mid-range t, which isolates the cause to the t -> noise-level mapping, not the weights.

**Root cause:** The network is conditioned on the step index t, so it implicitly learns abar_t of the training schedule. Cosine keeps more signal at each t, so the model over-denoises and contracts samples toward the mean.

**Repair:** Sample with the exact schedule used in training (the Part 1 linear schedule).

**Evidence (repaired):**
  - support_fraction: 0.7177
  - swd: 0.09056
  - mmd: 0.009416
  - sample_std: 1.094
  - metric_floor_swd: 0.01644

**Guard before scaling:** checkpoint stores schedule_fingerprint (sha1 of betas); ddpm.check_schedule raises on mismatch and on abar_T > 1e-3. Report SWD/MMD, not support fraction alone.

## F2: Exploding learning rate (Rectified flow (1-RF))

**Induced by:** Adam lr = 1.0 with no warmup (baseline: lr = 1e-3, 500-step warmup); second run adds grad_clip = 1.0.

**Symptom:** Loss and grad-norm spike in the first logging window, then a calm-looking plateau at the constant-predictor loss; no NaN, so nothing crashes. Samples collapse.

**Evidence (broken):**
  - peak_loss: 857
  - final_loss: 2.339
  - constant_predictor_loss: 2.322
  - peak_grad_norm: 338.5
  - pred_to_target_std: 0.1652
  - param_norm: 5426
  - support_fraction: 0
  - swd: 8.56
  - with_clip_final_loss: 2.331
  - with_clip_pred_to_target_std: 0.002778
  - with_clip_swd: 8.256

**Diagnosis:** LR range test shows a stable band and a cliff: above it the final loss equals the closed-form constant-predictor loss Var(x)+Var(z) and prediction spread -> 0. Gradient clipping at the same LR leaves the loss at 2.331, ruling out gradient magnitude as the lever.

**Root cause:** Adam's update is ~lr per parameter irrespective of gradient scale, so lr = 1.0 moves every weight by O(1) per step; weights blow up, activations saturate and the network becomes a constant function from which gradients cannot recover it.

**Repair:** lr = 1e-3 with 500-step linear warmup and cosine decay; range test: LRs that avoid collapse [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03], lowest final loss at 1e-02.

**Evidence (repaired):**
  - final_loss: 1.697
  - pred_to_target_std: 0.5211
  - param_norm: 52.48
  - support_fraction: 0.7838
  - swd: 0.05753
  - metric_floor_swd: 0.01644

**Guard before scaling:** LR range test before each new scale/architecture; abort a run whose loss stays within 5% of the closed-form constant-predictor loss; log parameter norm (103x baseline here) as a second alarm. Don't count on clipping with Adam.

## F3: Resolution mismatch between training and evaluation grids (FNO-1d)

**Induced by:** Coordinate channel built as grid index j instead of x = j/N; trained at N=128, evaluated at N = 32 .. 1024.

**Symptom:** Plausible error on the training grid (0.027) but 0.616 at N=1024 and 0.254 at N=64; no error raised.

**Evidence (broken):**
  - rel_l2_N32: 0.415
  - rel_l2_N64: 0.2545
  - rel_l2_N128: 0.02708
  - rel_l2_N256: 0.2812
  - rel_l2_N512: 0.4729
  - rel_l2_N1024: 0.6157
  - consistency_gap: 0.6127

**Diagnosis:** Error is minimal only at the training N and grows in both directions. Only the coordinate channel's values change with N (range [0, N-1]), so the pointwise lift sees out-of-distribution inputs; spectral layers are not the cause. Label-free resolution-consistency gap confirms it.

**Root cause:** Coordinates in index units make the input distribution resolution-dependent, breaking the discretization invariance that the Fourier layers otherwise provide.

**Repair:** Use physical coordinates x_j = j/N on the periodic unit interval (FNO1d.grid).

**Evidence (repaired):**
  - rel_l2_N32: 0.009064
  - rel_l2_N64: 0.008255
  - rel_l2_N128: 0.008244
  - rel_l2_N256: 0.008243
  - rel_l2_N512: 0.008244
  - rel_l2_N1024: 0.008244
  - consistency_gap: 0.0003815

**Guard before scaling:** resolution_consistency(model, u0_fine, coarse) < 1e-2 at every deployment grid, checked in CI; always report error at more than one grid.

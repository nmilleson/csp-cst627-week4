"""Decision walkthrough: turn the measured RESULTS dict into scorecards and a recommendation.

Nothing here is hardcoded to a particular run. Every number in the memo is read from
RESULTS (as written to results/results.json), and the recommendation follows explicit
rules, so the same code produces a correct memo from a CPU smoke test or a full GPU run.

Decision rules
--------------
Generative (DDPM vs 1-RF vs 2-RF, same data/backbone/budget):
  * quality target = SWD <= 2x the metric floor (two independent draws of the true data).
    If no recipe reaches it, the target falls back to 1.25x the best SWD any recipe
    achieved, and the memo says so.
  * for each recipe: the cheapest sampler setting (fewest NFE) that meets the target, and
    its latency per 1k samples. Pick = lowest latency at target; ties go to the cheaper
    training pipeline.
Operator (FNO vs the solver it would replace):
  * "pilot-ready" if the FNO is at least as accurate as the cheap classical baseline
    (coarse solver, within 10%) AND faster than it, AND the resolution-consistency guard
    passed in the failure study.
"""

SWD_TARGET_X_FLOOR = 2.0
FALLBACK_X_BEST = 1.25


def _per_1k_ms(row: dict, n_eval: int) -> float:
    return 1000.0 * row["latency_s"] / n_eval * 1000.0


def generative_rows(results: dict) -> dict[str, list[dict]]:
    """Sampler sweeps per recipe, sorted by NFE."""
    rows = {"DDPM": results["ddpm"]["sweep"],
            "1-RF": results["rf"]["rf1"]["sweep"],
            "2-RF": results["rf"]["rf2"]["sweep"]}
    return {k: sorted(v, key=lambda m: m["nfe"]) for k, v in rows.items()}


def quality_target(results: dict) -> tuple[float, str]:
    floor = results["ddpm"]["floor"]["swd"]
    target = SWD_TARGET_X_FLOOR * floor
    best = min(m["swd"] for rows in generative_rows(results).values() for m in rows)
    if best <= target:
        return target, f"SWD <= {target:.3f} ({SWD_TARGET_X_FLOOR:g}x the metric floor {floor:.3f})"
    target = FALLBACK_X_BEST * best
    return target, (f"SWD <= {target:.3f} (no recipe reached {SWD_TARGET_X_FLOOR:g}x floor = "
                    f"{SWD_TARGET_X_FLOOR * floor:.3f}; fallback is {FALLBACK_X_BEST}x the best SWD "
                    f"achieved, {best:.3f}; models are likely under-trained)")


def generative_scorecard(results: dict) -> list[dict]:
    n_eval = results["meta"]["n_eval"]
    target, _ = quality_target(results)
    train = {
        "DDPM": (results["ddpm"]["train_seconds"], results["ddpm"]["train_peak_vram_mb"]),
        "1-RF": (results["rf"]["rf1"]["train_seconds"], results["rf"]["rf1"]["train_peak_vram_mb"]),
        "2-RF": (results["rf"]["rf2"]["pipeline_seconds"], results["rf"]["rf2"]["train_peak_vram_mb"]),
    }
    card = []
    for name, rows in generative_rows(results).items():
        best = min(rows, key=lambda m: m["swd"])
        hit = next((m for m in rows if m["swd"] <= target), None)
        card.append({
            "recipe": name,
            "train_seconds": train[name][0],
            "train_peak_vram_mb": train[name][1],
            "best_swd": best["swd"], "best_support": best["support_fraction"], "best_at_nfe": best["nfe"],
            "nfe_at_target": hit["nfe"] if hit else None,
            "swd_at_target": hit["swd"] if hit else None,
            "ms_per_1k_at_target": _per_1k_ms(hit, n_eval) if hit else None,
            "sample_peak_vram_mb": hit.get("peak_vram_mb") if hit else None,
            "needs_data_bounds": name == "DDPM",   # x0 clipping (Part 1) needs known data range
        })
    return card


def operator_scorecard(results: dict) -> dict:
    fno, cost = results["fno"], results["fno"]["cost"]
    fine_ref = cost["solver @ N=1024"]["seconds"]
    res = {r["n_eval"]: r for r in fno["resolution_sweep"]}
    f3 = next((e for e in results.get("failure_log", []) if e["id"] == "F3"), None)
    return {
        "train_seconds": fno["train_seconds"], "train_peak_vram_mb": fno["train_peak_vram_mb"],
        "train_grid": fno["train_grid"],
        "rel_l2_mean_1024": res[1024]["mean"], "rel_l2_p95_1024": res[1024]["p95"],
        "rel_l2_max_1024": res[1024]["max"],
        "rel_l2_spread_across_grids": max(r["mean"] for r in res.values()) - min(r["mean"] for r in res.values()),
        "methods": {k: dict(v, speedup_vs_fine=fine_ref / v["seconds"]) for k, v in cost.items()},
        "consistency_gap": f3["evidence_repaired"]["consistency_gap"] if f3 else None,
    }


def recommend(results: dict) -> dict:
    gen = generative_scorecard(results)
    op = operator_scorecard(results)
    m = op["methods"]
    fno_fast, coarse = m[f"FNO @ N={op['train_grid']}"], m["solver @ N=128"]
    op_accurate = fno_fast["rel_l2"] <= 1.10 * coarse["rel_l2"]
    op_faster = fno_fast["seconds"] < coarse["seconds"]
    gap = op["consistency_gap"]
    op_consistent = gap is not None and gap < 1e-2

    reached = [g for g in gen if g["nfe_at_target"] is not None]
    gen_pick = (min(reached, key=lambda g: (g["ms_per_1k_at_target"], g["train_seconds"]))
                if reached else None)
    return {
        "operator_pilot": op_accurate and op_faster and op_consistent,
        "operator_checks": {"as_accurate_as_coarse_solver": op_accurate, "faster_than_coarse_solver": op_faster,
                            "resolution_consistent": op_consistent},
        "operator_speedup_vs_coarse": coarse["seconds"] / fno_fast["seconds"],
        "operator_speedup_vs_fine": fno_fast["speedup_vs_fine"],
        "fno_rel_l2": fno_fast["rel_l2"], "coarse_rel_l2": coarse["rel_l2"],
        "generative_pick": gen_pick["recipe"] if gen_pick else None,
        "generative_card": gen, "operator_card": op,
    }


# ----------------------------------------------------------------------------- formatting


def _f(v, spec=".3f", none="—"):
    return none if v is None else format(v, spec)


def generative_table_md(results: dict) -> str:
    target, desc = quality_target(results)
    lines = [f"Quality target: {desc}.", "",
             "| Recipe | Train wall-clock (s) | Train peak VRAM (MiB) | Best SWD (support) @ NFE "
             "| NFE to hit target | Latency at target (ms / 1k samples) | Needs known data bounds |",
             "|---|---|---|---|---|---|---|"]
    for g in generative_scorecard(results):
        lines.append(
            f"| {g['recipe']} | {g['train_seconds']:.0f} | {_f(g['train_peak_vram_mb'], '.0f', 'n/a (CPU)')} "
            f"| {g['best_swd']:.3f} ({g['best_support']:.3f}) @ {g['best_at_nfe']} "
            f"| {_f(g['nfe_at_target'], 'd', 'not reached')} | {_f(g['ms_per_1k_at_target'], '.1f')} "
            f"| {'yes (x0 clipping)' if g['needs_data_bounds'] else 'no'} |")
    lines.append("")
    lines.append("2-RF wall-clock includes training the 1-RF it distills from and generating its coupling pairs.")
    return "\n".join(lines)


def operator_table_md(results: dict) -> str:
    op = operator_scorecard(results)
    n_cases = results["meta"]["n_test_cases"]
    lines = ["| Method | Time for test set (s) | ms / case | Speedup vs fine solver | rel. L2 vs fine solver "
             "| Peak VRAM (MiB) |", "|---|---|---|---|---|---|"]
    for k, c in op["methods"].items():
        lines.append(f"| {k} | {c['seconds']:.4f} | {1000 * c['seconds'] / n_cases:.3f} | "
                     f"{c['speedup_vs_fine']:.0f}x | {c['rel_l2']:.4f} | "
                     f"{_f(c['peak_vram_mb'], '.0f', 'n/a (CPU)')} |")
    lines += ["", f"FNO trained at N={op['train_grid']} in {op['train_seconds']:.0f}s; at N=1024: mean rel. L2 "
              f"{op['rel_l2_mean_1024']:.4f}, 95th pct {op['rel_l2_p95_1024']:.4f}, worst {op['rel_l2_max_1024']:.4f}; "
              f"spread of mean error across N=32..1024: {op['rel_l2_spread_across_grids']:.4f}."]
    return "\n".join(lines)


def memo_md(results: dict, checks_md: str) -> str:
    meta = results["meta"]
    r = recommend(results)
    gen = {g["recipe"]: g for g in r["generative_card"]}
    hw = meta["device_name"] or meta["device"]
    caveat = ("\n> **These numbers come from a QUICK (reduced-budget, CPU) run. They are for "
              "checking the pipeline, not for the decision. Re-run on a GPU with QUICK off.**\n"
              if meta["quick"] else "")

    if r["operator_pilot"]:
        op_line = (f"**Pilot the Fourier Neural Operator.** On the Burgers benchmark its error against the fine "
                   f"solver is {r['fno_rel_l2']:.2%} relative L2, versus {r['coarse_rel_l2']:.2%} for the cheap "
                   f"classical solver, while running **{r['operator_speedup_vs_coarse']:.0f}x faster** than that solver "
                   f"and **{r['operator_speedup_vs_fine']:.0f}x faster** than the fine solver. It also passed the "
                   f"resolution-consistency guard, so one trained model serves every grid tested.")
    else:
        failed = [k.replace("_", " ") for k, ok in r["operator_checks"].items() if not ok]
        op_line = (f"**Do not pilot the neural operator yet.** It failed: {', '.join(failed)}. "
                   f"(FNO {r['fno_rel_l2']:.2%} vs coarse solver {r['coarse_rel_l2']:.2%} relative L2; "
                   f"{r['operator_speedup_vs_coarse']:.1f}x vs coarse solver.)")

    pick = r["generative_pick"]
    if pick:
        p = gen[pick]
        others = "; ".join(
            f"{k}: {g['ms_per_1k_at_target']:.1f} ms at NFE {g['nfe_at_target']}" if g["nfe_at_target"]
            else f"{k}: target not reached (best SWD {g['best_swd']:.3f})"
            for k, g in gen.items() if k != pick)
        bounds = ("" if pick == "DDPM" else " DDPM also depends on clipping to known data bounds, which real "
                  "wave-load fields won't provide.")
        gen_line = (f"**Of the generative recipes, {pick} is the one to carry forward.** It reaches the quality "
                    f"target with {p['nfe_at_target']} network evaluations, {p['ms_per_1k_at_target']:.1f} ms per "
                    f"1k samples (others: {others}).{bounds}")
    else:
        gen_line = "**No generative recipe reached the quality target**, so none is recommended yet."

    return f"""# Decision memo: generative and operator-learning pilot

**To:** Kwame Boateng · **Hardware measured on:** {hw} · **Mode:** {'QUICK smoke test' if meta['quick'] else 'full budget'}
{caveat}
## Recommendation

{op_line}

{gen_line}

The operator is the stronger pilot candidate for Halden because it directly replaces a cost we already pay, solver runs, and we measured that saving against the solver itself. The generative models were measured only against each other on a 2D toy distribution. They tell us which recipe to prefer, not yet whether either one earns money.

## Measured basis

### Operator: FNO vs. the solver it would replace
{operator_table_md(results)}

### Generative: DDPM vs. rectified flow (same data, network, training budget)
{generative_table_md(results)}

## Failure study

Three failures were induced, diagnosed and repaired. Each now has an automated guard (full log: `results/failure_log.md`):
{chr(10).join(f"- **{e['id']} ({e['model']}): {e['title']}.** Guard: {e['guard']}" for e in results.get('failure_log', []))}

## What must be checked before scaling

{checks_md}
"""

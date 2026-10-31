"""Human-readable status, measurement-noise controls, figures and the summary report."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .solver import success
from .store import Store


def _clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def save_json(path, obj) -> None:
    with open(path, "w") as fh:
        json.dump(_clean(obj), fh, indent=1)


# ------------------------------------------------------------------ store status ------
def noise_controls(records: List[Dict], mode: str = "unsat") -> Dict:
    """Repeat runs of the UNCHANGED original query: how stable is the measurement itself?"""
    by_q: Dict[str, List[Dict]] = {}
    for r in records:
        if r["variant"] == "orig:0":
            by_q.setdefault(r["query"], []).append(r)
    multi = {q: v for q, v in by_q.items() if len(v) >= 2}
    flips, rel = 0, []
    for q, v in multi.items():
        st = {success(r["outcome"], mode) for r in v if r["outcome"] != "error"}
        flips += len(st) > 1
        ok = [r["cost_s"] for r in v if success(r["outcome"], mode)]
        if len(ok) >= 2 and min(ok) > 0.05:
            rel.append((max(ok) - min(ok)) / np.mean(ok))
    return {
        "queries_with_repeats": len(multi),
        "outcome_flips_on_identical_input": flips,
        "median_rel_runtime_spread": float(np.median(rel)) if rel else None,
        "p95_rel_runtime_spread": float(np.percentile(rel, 95)) if rel else None,
    }


def status_text(root) -> str:
    store = Store(root)
    meta = store.load_json("meta.json", {})
    manifest = store.load_json("queries.json", [])
    recs = store.read()
    lines = [f"folder: {root}"]
    if meta.get("synthetic"):
        lines.append("*** SYNTHETIC mock-solver data: pipeline test only, not a scientific result ***")
    lines.append(f"solver: {meta.get('solver_id')}   timeout: {meta.get('timeout_s')} s   families: {meta.get('families')} x {meta.get('variants_per_family')}")
    lines.append(f"queries in manifest: {len(manifest)}   runs recorded: {len(recs)}")
    latest = store.latest()
    oc = Counter(r["outcome"] for k, r in latest.items() if k[2] == 0)
    lines.append("outcomes (repeat 0): " + ", ".join(f"{k}={v}" for k, v in sorted(oc.items())))
    cost = sum(r["cost_s"] for r in recs)
    lines.append(f"total solver cost: {cost / 3600:.2f} CPU hours (timeouts charged in full)")
    n_err = oc.get("error", 0)
    if n_err:
        ex = [r for r in latest.values() if r["outcome"] == "error"][:3]
        lines.append(f"WARNING: {n_err} runs ended in 'error' (not counted as failures). Examples: "
                     + "; ".join(f"{r['query']} {r['variant']}" for r in ex))
        lines.append("  Errors on EVERY run mean a solver-flag problem (run `python -m fragile check`); "
                     "errors only on some variants of a query mean a bad variant - report it.")
    loads = [r["load1"] for r in recs if r.get("load1", -1) >= 0]
    if loads:
        lines.append(f"machine load while measuring: median {np.median(loads):.1f}, p95 {np.percentile(loads, 95):.1f} "
                     f"(keep this below the number of workers + 1; high values add noise)")
    nc = noise_controls(recs, meta.get("success_mode", "unsat"))
    lines.append(f"noise controls: {nc['queries_with_repeats']} queries re-run unchanged; "
                 f"{nc['outcome_flips_on_identical_input']} changed outcome with IDENTICAL input; "
                 f"median runtime spread {nc['median_rel_runtime_spread']}")
    by_q = Counter(k[0] for k in latest if k[2] == 0)
    complete = sum(1 for m in manifest if by_q.get(m["qid"], 0) >= len(m["variants"]))
    lines.append(f"complete queries: {complete}/{len(manifest)}")
    return "\n".join(lines)


# ------------------------------------------------------------------ figures -----------
COLORS = {"uniform": "#555555", "random": "#aaaaaa", "runtime": "#d98e04", "adaptive": "#1f77b4",
          "adaptive-tuned": "#d62728"}


def make_figures(res: Dict, outdir: Path, synthetic: bool) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    frac = np.array(res["frac"])
    hours = np.array(res["hours"])
    paths = []
    mark = "SYNTHETIC DATA - PIPELINE TEST ONLY\n" if synthetic else ""

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, s in res["strategies"].items():
        c = COLORS.get(name)
        ax[0].plot(hours, np.array(s["curve_mean"]), label=name, color=c, lw=2)
        ax[0].fill_between(hours, s["curve_lo"], s["curve_hi"], color=c, alpha=0.12, lw=0)
        ax[1].plot(hours, np.array(s["false_alarm_count_mean"]), label=name, color=c, lw=2)
    ax[0].set_xlabel("total CPU time spent testing (hours)")
    ax[0].set_ylabel("fraction of detectable fragile queries found")
    ax[0].set_title(mark + "Detected fragile queries vs CPU time (held-out projects)")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("total CPU time spent testing (hours)")
    ax[1].set_ylabel("queries flagged that look stable in the independent reference")
    ax[1].set_title(mark + "Flags on queries the reference calls pure")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig1_detection_vs_cpu.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(str(p))

    comps = res["comparisons"]
    if comps:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        names = list(comps)
        mean = [comps[n]["audc_diff_vs_uniform_mean"] for n in names]
        lo = [comps[n]["audc_diff_seed_p2.5"] for n in names]
        hi = [comps[n]["audc_diff_seed_p97.5"] for n in names]
        ax.barh(names, mean, color=[COLORS.get(n, "#888") for n in names])
        ax.errorbar(mean, names, xerr=[np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)], fmt="none", ecolor="k", capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("AUDC difference vs uniform (bars: 2.5-97.5% over seeds)")
        ax.set_title(mark + "Paired comparison")
        fig.tight_layout()
        p = outdir / "fig2_audc_difference.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(str(p))
    return paths


# ------------------------------------------------------------------ summary -----------
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def summary_markdown(res: Dict, grid_report: Dict, meta: Dict, noise: Dict, figs: List[str]) -> str:
    syn = bool(meta.get("synthetic"))
    L = []
    if syn:
        L += ["> **SYNTHETIC DATA.** These numbers come from the mock solver and test only that the code runs.",
              "> They are not evidence about any real solver and must never be reported as results.", ""]
    L += ["# Fragile-proof detection: summary", ""]
    L += [f"- solver: `{meta.get('solver_id')}`; timeout {meta.get('timeout_s')} s; success = `{meta.get('success_mode', 'unsat')}`",
          f"- queries analysed: {res['data']['queries']} in {res['settings']['n_projects']} projects "
          f"(manifest {grid_report.get('queries_in_manifest')}, incomplete {grid_report.get('incomplete')}, "
          f"dropped for original-query error {grid_report.get('orig_error')}, variant errors skipped {grid_report.get('variant_errors')})",
          f"- labels from independent reference variants (fragile = success rate in [{res['settings']['tau']}, {1 - res['settings']['tau']}]): "
          f"{res['data']['labels']}",
          f"- detectable from the pool at all (the ceiling): {res['data']['flaggable_by_label']}",
          f"- exhaustive pool testing would cost {res['data']['pool_total_cpu_hours']:.2f} CPU hours (the x-axis scale)",
          f"- seeds {res['settings']['seeds']}, project-held-out folds {res['settings']['folds']}, tuned: {res['settings']['tune']}", ""]
    n_frag = res["data"]["flaggable_by_label"]["fragile"]
    if n_frag < 30:
        L += [f"> **UNDERPOWERED:** only {n_frag:.0f} detectable fragile queries. Differences between strategies at this size are "
              "mostly noise; collect more queries (or report this as a pilot) before drawing any conclusion.", ""]
    L += ["## Main result", "", "| strategy | AUDC (mean, 95% over seeds) | budget to 50% of ceiling | budget to 80% | fragile found @10% / @25% budget | false alarms @25% |", "|---|---|---|---|---|---|"]
    for n, s in res["strategies"].items():
        a10, a25 = s["at_budget"]["10%"], s["at_budget"]["25%"]
        L.append(f"| {n} | {_f(s['audc_mean'])} [{_f(s['audc_p2.5'])}, {_f(s['audc_p97.5'])}] | {_f(s['budget_to_50pct_of_ceiling'], 2)} | "
                 f"{_f(s['budget_to_80pct_of_ceiling'], 2)} | {_f(a10['fragile_found_frac_of_ceiling'], 2)} / {_f(a25['fragile_found_frac_of_ceiling'], 2)} | {_f(a25['false_alarms_pure'], 1)} |")
    L += ["", "Budgets are fractions of the cost of testing every pool variant of every query (1.0 = exhaustive).", ""]
    L += ["## Paired comparison with uniform testing", "", "| strategy | AUDC difference | 95% over seeds | seeds better | 95% over projects |", "|---|---|---|---|---|"]
    for n, c in res["comparisons"].items():
        pb = (f"[{_f(c['project_bootstrap_p2_5'])}, {_f(c['project_bootstrap_p97_5'])}]" if "project_bootstrap_p2_5" in c else c.get("project_bootstrap_note", "n/a"))
        L.append(f"| {n} | {_f(c['audc_diff_vs_uniform_mean'], 4)} | [{_f(c['audc_diff_seed_p2.5'], 4)}, {_f(c['audc_diff_seed_p97.5'], 4)}] | "
                 f"{c['seeds_better_than_uniform']}/{res['settings']['seeds']} | {pb} |")
    L += ["", "Seeds capture variation of test order and strategy randomness; the project interval captures which projects "
          "were sampled (approximate: allocation was not re-run on each resample).", ""]
    if res.get("group_crosstab"):
        L += ["## Agreement with the benchmark's curated classification", "",
              "Rows: the benchmark's own group (curated by its authors with a different solver build and timeout).",
              "Columns: how THIS experiment's independent reference labels those queries.", "",
              "| curated group | pure | borderline | fragile | fragile share |", "|---|---|---|---|---|"]
        for g, c in sorted(res["group_crosstab"].items()):
            tot = sum(c.values())
            L.append(f"| {g or '(none)'} | {c['pure']} | {c['borderline']} | {c['fragile']} | {c['fragile'] / tot:.0%} |")
        L += ["", "If curated-unstable groups show a clearly higher fragile share than curated-stable groups, the labelling and "
              "the variant generator are behaving sensibly on real data. If not, investigate before trusting any strategy comparison.", ""]
    if res["tuned_params"]:
        L += ["## Parameters chosen on training projects", ""] + [f"- fold {t['fold'] + 1}: {t['params']}" for t in res["tuned_params"]] + [""]
    L += ["## Measurement controls", "",
          f"- {noise['queries_with_repeats']} unchanged originals were re-run; {noise['outcome_flips_on_identical_input']} changed outcome with identical input "
          f"(these are timing/limit noise, not input sensitivity); median relative runtime spread {_f(noise['median_rel_runtime_spread'], 3)}.",
          f"- queries with both `sat` and `unsat` among equivalent variants (soundness-bug candidates OR a transformation bug - investigate before claiming anything): "
          f"{len(grid_report.get('conflicts', []))} {grid_report.get('conflicts', [])[:5]}", ""]
    if figs:
        L += ["## Figures", ""] + [f"- {Path(p).name}" for p in figs] + [""]
    L += ["## How to read this honestly", "",
          "- A strategy only 'wins' if its curve is above uniform on held-out projects with intervals that exclude zero.",
          "- Witnesses found here are measurements; confirm them with `python -m fragile confirm` (re-runs) before calling a query fragile.",
          "- Nothing here shows that unflagged queries are stable; the ceiling is limited by the pool size.", ""]
    return "\n".join(L)

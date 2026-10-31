"""Evaluate allocation strategies by replay.

Protocol (fixed in advance, see README):
  * labels come from independent reference variants (odd-index variants), strategies only see the pool;
  * all strategies share stopping rules and per-seed variant orders (paired design);
  * the adaptive strategy's hyper-parameters are tuned on TRAINING projects and evaluated on
    HELD-OUT projects (K-fold over projects) - queries of a project are never split;
  * headline metric: detected fragile queries vs CPU budget, summarised by the area under the
    normalised curve (AUDC) and the budget needed to reach 50% / 80% of what exhaustive pool testing finds.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np

from .grid import BORDER, FRAGILE, PURE, Grid
from .simulate import audc, budget_to_reach, ceilings, detection_counts, run_sim
from .strategies import REGISTRY

TUNE_GRIDS = {
    "small": {"eps": [0.05, 0.2], "gamma": [0.0, 3.0], "kappa": [5.0, 25.0]},
    "full": {"eps": [0.0, 0.05, 0.1, 0.25], "gamma": [0.0, 1.0, 3.0, 8.0], "kappa": [2.0, 10.0, 40.0], "pi0": [0.02, 0.1]},
}
DEFAULT_STRATEGIES = ["uniform", "random", "runtime", "adaptive", "adaptive-tuned"]
CHECKPOINTS = (0.05, 0.10, 0.25, 0.50)


def _pooled_fragile(counts, ceil):
    """counts [seeds,P,3,nb] -> normalised pooled fragile curve [seeds,nb]."""
    den = ceil[:, FRAGILE].sum()
    return counts[:, :, FRAGILE, :].sum(axis=1) / den


def _tune(train: Grid, tau, label_mode, grid_name, tune_seeds, frac):
    labels = train.labels(tau, label_mode)
    ceil = ceilings(train, labels)
    if ceil[:, FRAGILE].sum() == 0:
        return {}, []
    space = TUNE_GRIDS[grid_name]
    keys = list(space)
    cls = REGISTRY["adaptive"][0]
    best, best_val, table = {}, -1.0, []
    for combo in itertools.product(*(space[k] for k in keys)):
        params = dict(zip(keys, combo))
        vals = []
        for s in range(tune_seeds):
            ev_c, ev_q, _ = run_sim(train, cls, params, 10_000 + s)
            cnt = detection_counts(train, labels, ev_c, ev_q, frac)
            vals.append(audc(cnt[:, FRAGILE, :].sum(0) / ceil[:, FRAGILE].sum(), frac))
        v = float(np.mean(vals))
        table.append({"params": params, "train_audc": v})
        if v > best_val:
            best, best_val = params, v
    return best, table


def evaluate(
    grid: Grid,
    seeds: int = 20,
    folds: int = 3,
    tau: float = 0.1,
    label_mode: str = "independent",
    strategies: Optional[List[str]] = None,
    tune: bool = True,
    tune_seeds: int = 3,
    tune_grid: str = "small",
    nb: int = 201,
    boot: int = 2000,
    rng_seed: int = 0,
    log=print,
) -> Dict:
    strategies = list(strategies or DEFAULT_STRATEGIES)
    if not tune:
        strategies = [s for s in strategies if s != "adaptive-tuned"]
    frac = np.linspace(0.0, 1.0, nb)
    P = len(grid.proj_names)
    k = min(folds, P) if P >= 2 else 1
    rng = np.random.default_rng(rng_seed)
    fold_of = np.zeros(P, dtype=int)
    for i, p in enumerate(rng.permutation(P)):
        fold_of[p] = i % k
    can_tune = tune and k >= 2
    if tune and not can_tune:
        log("only one project: cannot tune on held-out projects; 'adaptive-tuned' is skipped")
        strategies = [s for s in strategies if s != "adaptive-tuned"]

    counts = {s: np.zeros((seeds, P, 3, nb)) for s in strategies}
    ceil_all = np.zeros((P, 3))
    n_by_label = np.zeros((P, 3))
    total_cost = 0.0
    tuned_params = []
    for fold in range(k):
        test_p = [p for p in range(P) if fold_of[p] == fold]
        train_p = [p for p in range(P) if fold_of[p] != fold]
        test = grid.subset(test_p)
        lab = test.labels(tau, label_mode)
        ceil = ceilings(test, lab)
        total_cost += test.pool_total_cost()
        full_idx = sorted(test_p)
        ceil_all[full_idx] = ceil
        for q in range(test.n):
            n_by_label[full_idx[int(test.proj_idx[q])], lab[q]] += 1
        best = {}
        if "adaptive-tuned" in strategies and train_p:
            best, table = _tune(grid.subset(train_p), tau, label_mode, tune_grid, tune_seeds, frac)
            tuned_params.append({"fold": fold, "test_projects": [grid.proj_names[p] for p in test_p], "params": best})
            log(f"fold {fold + 1}/{k}: tuned on {len(train_p)} projects -> {best}")
        else:
            log(f"fold {fold + 1}/{k}: {len(test_p)} held-out projects")
        for name in strategies:
            base = "adaptive" if name == "adaptive-tuned" else name
            cls, params = REGISTRY[base]
            params = dict(params)
            if name == "adaptive-tuned":
                params.update(best)
            for s in range(seeds):
                ev_c, ev_q, _ = run_sim(test, cls, params, s)
                counts[name][s][full_idx] = detection_counts(test, lab, ev_c, ev_q, frac)

    if ceil_all[:, FRAGILE].sum() == 0:
        raise SystemExit(
            "No fragile query can be detected from the pool (none is flaggable). Collect more/other queries "
            "or a larger --timeout; there is nothing to compare strategies on."
        )
    den = ceil_all[:, FRAGILE].sum()
    hours = frac * total_cost / 3600.0
    out: Dict = {
        "settings": {"seeds": seeds, "folds": k, "tau": tau, "label_mode": label_mode, "tune": can_tune,
                     "tune_grid": tune_grid if can_tune else None, "n_projects": P},
        "data": {
            "queries": grid.n,
            "projects": grid.proj_names,
            "labels": {"pure": float(n_by_label[:, PURE].sum()), "borderline": float(n_by_label[:, BORDER].sum()),
                       "fragile": float(n_by_label[:, FRAGILE].sum())},
            "flaggable_by_label": {"pure": float(ceil_all[:, PURE].sum()), "borderline": float(ceil_all[:, BORDER].sum()),
                                   "fragile": float(den)},
            "pool_total_cpu_hours": total_cost / 3600.0,
        },
        "tuned_params": tuned_params,
        "frac": frac.tolist(),
        "hours": hours.tolist(),
        "strategies": {},
        "comparisons": {},
    }
    per_seed_audc = {}
    mean_counts = {}
    for name in strategies:
        y = _pooled_fragile(counts[name], ceil_all)  # seeds x nb
        a = np.array([audc(y[s], frac) for s in range(seeds)])
        per_seed_audc[name] = a
        mean_counts[name] = counts[name][:, :, FRAGILE, :].mean(axis=0)  # P x nb
        fa = counts[name][:, :, PURE, :].sum(axis=1)
        fr = counts[name][:, :, FRAGILE, :].sum(axis=1)
        bd = counts[name][:, :, BORDER, :].sum(axis=1)
        pts = {}
        for c in CHECKPOINTS:
            i = int(round(c * (nb - 1)))
            pts[f"{int(c * 100)}%"] = {
                "fragile_found_frac_of_ceiling": float(y[:, i].mean()),
                "fragile_found": float(fr[:, i].mean()),
                "false_alarms_pure": float(fa[:, i].mean()),
                "borderline_flagged": float(bd[:, i].mean()),
            }
        out["strategies"][name] = {
            "audc_mean": float(a.mean()),
            "audc_sd": float(a.std(ddof=1)) if seeds > 1 else 0.0,
            "audc_p2.5": float(np.percentile(a, 2.5)),
            "audc_p97.5": float(np.percentile(a, 97.5)),
            "budget_to_50pct_of_ceiling": _nanmean([budget_to_reach(y[s], frac, 0.5) for s in range(seeds)]),
            "budget_to_80pct_of_ceiling": _nanmean([budget_to_reach(y[s], frac, 0.8) for s in range(seeds)]),
            "at_budget": pts,
            "curve_mean": y.mean(axis=0).tolist(),
            "curve_lo": np.percentile(y, 2.5, axis=0).tolist(),
            "curve_hi": np.percentile(y, 97.5, axis=0).tolist(),
            "fragile_count_mean": fr.mean(axis=0).tolist(),
            "false_alarm_count_mean": fa.mean(axis=0).tolist(),
        }
    brng = np.random.default_rng(rng_seed + 1)
    for name in strategies:
        if name == "uniform":
            continue
        d = per_seed_audc[name] - per_seed_audc["uniform"]
        comp = {
            "audc_diff_vs_uniform_mean": float(d.mean()),
            "audc_diff_seed_p2.5": float(np.percentile(d, 2.5)),
            "audc_diff_seed_p97.5": float(np.percentile(d, 97.5)),
            "seeds_better_than_uniform": int((d > 0).sum()),
        }
        if P >= 8 and boot > 0:
            W = brng.multinomial(P, np.ones(P) / P, size=boot)
            num = W @ (mean_counts[name] - mean_counts["uniform"])
            dn = W @ ceil_all[:, FRAGILE]
            ok = dn > 0
            vals = audc_rows(num[ok] / dn[ok, None], frac)
            comp.update(
                project_bootstrap_p2_5=float(np.percentile(vals, 2.5)),
                project_bootstrap_p97_5=float(np.percentile(vals, 97.5)),
                project_bootstrap_reps=int(ok.sum()),
            )
        else:
            comp["project_bootstrap_note"] = f"only {P} projects: project-level CI not meaningful (need >= 8)"
        out["comparisons"][name] = comp
    return out


def audc_rows(Y: np.ndarray, frac: np.ndarray) -> np.ndarray:
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return trapz(Y, frac, axis=1)


def _nanmean(xs):
    a = np.array(xs, dtype=float)
    if np.isnan(a).all():
        return float("nan")
    return float(np.nanmean(a)) if (~np.isnan(a)).sum() >= len(a) / 2 else float("nan")

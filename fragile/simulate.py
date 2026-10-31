"""Replay a strategy on recorded measurements (no solver needed)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .grid import Grid
from .strategies import REGISTRY

_trapz = getattr(np, "trapezoid", None) or np.trapz


def run_sim(grid: Grid, cls, params: Dict, seed: int, budget: Optional[float] = None):
    """Run one strategy on the grid.  Returns (event_cost, event_query, total_spent).

    The per-query variant order depends only on ``seed`` (orig first, then a random order), so
    different strategies see identical data for the same seed (paired comparison).
    """
    rng_order = np.random.default_rng([seed, 1])
    rng_strat = np.random.default_rng([seed, 2])
    Q = grid.n
    succ, cost = [], []
    for q in range(Q):
        m = len(grid.pool_succ[q])
        perm = np.concatenate(([0], 1 + rng_order.permutation(m - 1))) if m > 1 else np.array([0])
        succ.append(grid.pool_succ[q][perm])
        cost.append(grid.pool_cost[q][perm])
    sizes = [len(s) for s in succ]
    strat = cls(sizes, grid.proj_idx, len(grid.proj_names), grid.timeout_s, rng_strat, **params)
    ptr = np.zeros(Q, dtype=np.int64)
    limit = grid.pool_total_cost() if budget is None else budget
    spent = 0.0
    ev_c, ev_q = [], []
    while spent < limit:
        q = strat.pick()
        if q is None:
            break
        i = int(ptr[q])
        ptr[q] += 1
        c = float(cost[q][i])
        spent += c
        was = bool(strat.flagged[q])
        strat.update(q, bool(succ[q][i]), c)
        if strat.flagged[q] and not was:
            ev_c.append(spent)
            ev_q.append(q)
    return np.array(ev_c), np.array(ev_q, dtype=np.int64), spent


def detection_counts(grid: Grid, labels: np.ndarray, ev_c, ev_q, frac: np.ndarray) -> np.ndarray:
    """counts[project, label, budget_point]: queries flagged with spend <= budget.

    ``frac`` is the budget as a fraction of the cost of testing every pool variant of every query.
    """
    total = grid.pool_total_cost()
    budgets = frac * total
    out = np.zeros((len(grid.proj_names), 3, len(frac)))
    if len(ev_c):
        pos = np.searchsorted(budgets, ev_c, side="left")
        ok = pos < len(frac)
        np.add.at(out, (grid.proj_idx[ev_q[ok]], labels[ev_q[ok]], pos[ok]), 1)
    return np.cumsum(out, axis=2)


def ceilings(grid: Grid, labels: np.ndarray) -> np.ndarray:
    """ceil[project, label]: how many queries of each label are flaggable at all from the pool."""
    flg = grid.flaggable()
    out = np.zeros((len(grid.proj_names), 3))
    for q in range(grid.n):
        if flg[q]:
            out[grid.proj_idx[q], labels[q]] += 1
    return out


def audc(y: np.ndarray, frac: np.ndarray) -> float:
    """Area under a normalised detection curve (0..1; higher = finds fragile queries earlier)."""
    return float(_trapz(y, frac))


def budget_to_reach(y: np.ndarray, frac: np.ndarray, target: float) -> float:
    idx = np.where(y >= target - 1e-12)[0]
    return float(frac[idx[0]]) if len(idx) else float("nan")

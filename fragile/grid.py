"""Turn raw observations into arrays for replay experiments.

For every query we keep
  * the POOL    : orig + variants with even index. Strategies may only test these.
  * the REF set : variants with odd index. Used only to label queries (independent of the pool).
Runs that ended in 'error' are NOT failures (they usually mean a broken variant or a bad solver
flag), so such variants are dropped and counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .solver import success
from .store import Store
from .variants import ORIG, pool_or_ref

PURE, BORDER, FRAGILE = 0, 1, 2
LABEL_NAMES = {PURE: "pure", BORDER: "borderline", FRAGILE: "fragile"}


@dataclass
class Grid:
    qids: List[str]
    proj_names: List[str]
    proj_idx: np.ndarray
    pool_succ: List[np.ndarray]  # bool, index 0 is the original query
    pool_cost: List[np.ndarray]
    ref_succ: List[np.ndarray]
    timeout_s: float
    meta: Dict = field(default_factory=dict)
    report: Dict = field(default_factory=dict)
    groups: List[str] = field(default_factory=list)  # optional curated group per query ("" if unknown)

    def __post_init__(self):
        if not self.groups:
            self.groups = [""] * len(self.qids)

    @property
    def n(self) -> int:
        return len(self.qids)

    def subset(self, project_ids: List[int]) -> "Grid":
        keep = [q for q in range(self.n) if int(self.proj_idx[q]) in set(project_ids)]
        remap = {p: i for i, p in enumerate(sorted(set(project_ids)))}
        return Grid(
            qids=[self.qids[q] for q in keep],
            proj_names=[self.proj_names[p] for p in sorted(set(project_ids))],
            proj_idx=np.array([remap[int(self.proj_idx[q])] for q in keep], dtype=np.int64),
            pool_succ=[self.pool_succ[q] for q in keep],
            pool_cost=[self.pool_cost[q] for q in keep],
            ref_succ=[self.ref_succ[q] for q in keep],
            timeout_s=self.timeout_s,
            meta=self.meta,
            report=self.report,
            groups=[self.groups[q] for q in keep],
        )

    def pool_total_cost(self) -> float:
        return float(sum(c.sum() for c in self.pool_cost))

    def labels(self, tau: float = 0.1, mode: str = "independent") -> np.ndarray:
        """0 pure (reference success rate 0 or 1), 1 borderline (inside (0,tau) or (1-tau,1)),
        2 fragile (rate in [tau, 1-tau]).  ``independent`` uses only the held-out reference
        variants; ``full`` also uses the pool (optimistic, for sensitivity analysis)."""
        lab = np.zeros(self.n, dtype=np.int64)
        for q in range(self.n):
            s = self.ref_succ[q] if mode == "independent" else np.concatenate([self.ref_succ[q], self.pool_succ[q]])
            r = float(s.mean())
            if r in (0.0, 1.0):
                lab[q] = PURE
            elif tau <= r <= 1 - tau:
                lab[q] = FRAGILE
            else:
                lab[q] = BORDER
        return lab

    def flaggable(self) -> np.ndarray:
        """True where testing the whole pool would reveal both a success and a failure."""
        return np.array([bool(s.any() and (~s).any()) for s in self.pool_succ])


def load_grid(root, min_pool: int = 4, min_ref: int = 4, success_mode: Optional[str] = None) -> Grid:
    store = Store(root)
    meta = store.load_json("meta.json", {})
    manifest = store.load_json("queries.json", [])
    if not manifest:
        raise SystemExit(f"{root}: no queries.json - run `collect` first")
    mode = success_mode or meta.get("success_mode", "unsat")
    obs: Dict[str, Dict[str, tuple]] = {}
    for r in store.read():
        if r["repeat"] == 0:
            obs.setdefault(r["query"], {})[r["variant"]] = (r["outcome"], float(r["cost_s"]))

    rep = {"queries_in_manifest": len(manifest), "incomplete": 0, "orig_error": 0, "too_few_variants": 0,
           "variant_errors": 0, "conflicts": []}
    rows = []
    for m in manifest:
        o = obs.get(m["qid"], {})
        if any(v not in o for v in m["variants"]):
            rep["incomplete"] += 1
            continue
        if o[ORIG][0] == "error":
            rep["orig_error"] += 1
            continue
        ps, pc, rs = [], [], []
        outcomes = set()
        for vid in m["variants"]:
            oc, cost = o[vid]
            if oc == "error":
                rep["variant_errors"] += 1
                continue
            outcomes.add(oc)
            if pool_or_ref(vid) == "pool":
                ps.append(success(oc, mode))
                pc.append(cost)
            else:
                rs.append(success(oc, mode))
        if "sat" in outcomes and "unsat" in outcomes:
            rep["conflicts"].append(m["qid"])
        if len(ps) < min_pool or len(rs) < min_ref:
            rep["too_few_variants"] += 1
            continue
        rows.append((m["qid"], m["project"], np.array(ps, bool), np.array(pc, float), np.array(rs, bool), m.get("group", "") or ""))
    if not rows:
        raise SystemExit("No complete queries to analyse yet (see `python -m fragile status`).")
    names = sorted({r[1] for r in rows})
    idx = {n: i for i, n in enumerate(names)}
    rep["queries_used"] = len(rows)
    return Grid(
        qids=[r[0] for r in rows],
        proj_names=names,
        proj_idx=np.array([idx[r[1]] for r in rows], dtype=np.int64),
        pool_succ=[r[2] for r in rows],
        pool_cost=[r[3] for r in rows],
        ref_succ=[r[4] for r in rows],
        timeout_s=float(meta.get("timeout_s", 10.0)),
        meta=meta,
        report=rep,
        groups=[r[5] for r in rows],
    )

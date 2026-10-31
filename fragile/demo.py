"""Generate SYNTHETIC demo queries.

The files are valid, easy, unsatisfiable SMT-LIB scripts (a real Z3 proves them instantly) with a
``; mock-params`` header that only the MOCK solver reads.  The hidden parameters deliberately make
some queries "mixed" (runtime distribution straddling the timeout) and cluster them in some
projects - an ASSUMPTION used to exercise the code, not an observation about real verifiers.
"""
from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

_N = statistics.NormalDist()


def _query_text(rng: random.Random, mu: float, sigma: float, p_unknown: float, n_aux: int) -> str:
    L = [f"; mock-params mu={mu:.4f} sigma={sigma:.4f} p_unknown={p_unknown:.3f}", "(set-logic UFLIA)"]
    L += ["(declare-fun f (Int) Int)", "(declare-const c Int)"]
    for i in range(n_aux):
        L.append(f"(declare-fun h{i} (Int) Int)")
    asserts = ["(forall ((x Int)) (>= (f x) x))", "(< (f c) c)"]
    for i in range(n_aux):
        kind = rng.randrange(3)
        if kind == 0:
            asserts.append(f"(forall ((x Int)) (>= (h{i} x) {rng.randrange(0, 5)}))")
        elif kind == 1:
            asserts.append(f"(forall ((x Int) (y Int)) (=> (> x y) (>= (h{i} x) (h{i} y))))")
        else:
            asserts.append(f"(= (h{i} c) (+ (f c) {rng.randrange(1, 9)}))")
    rng.shuffle(asserts)
    L += [f"(assert {a})" for a in asserts]
    L.append("(check-sat)")
    return "\n".join(L) + "\n"


def make_demo(out_dir, n_projects: int = 8, per_project: int = 15, timeout_s: float = 10.0, seed: int = 0) -> int:
    rng = random.Random(seed)
    out = Path(out_dir)
    lt = math.log(timeout_s)
    count = 0
    for p in range(n_projects):
        p_mixed = rng.choice([0.0, 0.03, 0.08, 0.2, 0.35])
        p_hopeless = rng.choice([0.05, 0.2, 0.4])
        d = out / f"project{p:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_project):
            u = rng.random()
            if u < p_mixed:  # mixed: success probability roughly 0.12 .. 0.88
                sigma = rng.uniform(0.8, 2.0)
                mu = lt + rng.uniform(-1.2, 1.2) * sigma
                pu = rng.choice([0.0, 0.1])
            elif u < p_mixed + 0.05:  # borderline: rarely flips
                sigma = rng.uniform(0.8, 1.5)
                mu = lt + rng.choice([-1, 1]) * rng.uniform(1.8, 2.6) * sigma
                pu = 0.0
            elif u < p_mixed + 0.05 + p_hopeless:
                sigma, mu, pu = rng.uniform(0.1, 0.4), lt + rng.uniform(3, 6), 0.0
            else:
                sigma, mu, pu = rng.uniform(0.1, 0.5), lt - rng.uniform(2.5, 6), 0.0
            text = _query_text(rng, mu, sigma, pu, n_aux=rng.randrange(4, 9))
            (d / f"q{i:03d}.smt2").write_text(text)
            count += 1
    return count

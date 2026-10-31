import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fragile.grid import FRAGILE, Grid
from fragile.simulate import audc, ceilings, detection_counts, run_sim
from fragile.strategies import REGISTRY, Adaptive


def synthetic_grid(seed=0, n_proj=4, per_proj=25, pool=8, timeout=10.0) -> Grid:
    rng = np.random.default_rng(seed)
    qids, projs, ps, pc, rs = [], [], [], [], []
    for p in range(n_proj):
        p_mixed = [0.02, 0.1, 0.25, 0.4][p % 4]
        for i in range(per_proj):
            theta = rng.uniform(0.15, 0.85) if rng.random() < p_mixed else float(rng.integers(0, 2))
            mean_cost = 2.0 if 0 < theta < 1 else rng.choice([0.05, 10.0])
            s = rng.random(pool) < theta
            c = np.where(s, rng.uniform(0.2, 1.0, pool) * mean_cost, timeout)
            qids.append(f"p{p}/q{i}")
            projs.append(p)
            ps.append(s)
            pc.append(c)
            rs.append(rng.random(pool) < theta)
    return Grid(qids, [f"p{p}" for p in range(n_proj)], np.array(projs), ps, pc, rs, timeout)


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.g = synthetic_grid()
        self.labels = self.g.labels()
        self.flag = self.g.flaggable()

    def test_full_budget_reaches_ceiling_for_every_strategy(self):
        for name in ("uniform", "random", "runtime", "adaptive", "adaptive-nopool", "adaptive-noexplore"):
            cls, params = REGISTRY[name]
            ev_c, ev_q, spent = run_sim(self.g, cls, params, seed=3)
            self.assertEqual(set(ev_q.tolist()), set(np.where(self.flag)[0].tolist()), name)
            self.assertEqual(len(ev_q), len(set(ev_q.tolist())), name)  # nobody flagged twice
            self.assertLessEqual(spent, self.g.pool_total_cost() + 1e-9, name)
            self.assertTrue(np.all(np.diff(ev_c) >= 0), name)

    def test_never_tests_more_than_pool_or_after_flag(self):
        rng = np.random.default_rng(1)
        for name in ("uniform", "random", "runtime", "adaptive"):
            cls, params = REGISTRY[name]
            sizes = [len(s) for s in self.g.pool_succ]
            strat = cls(sizes, self.g.proj_idx, 4, 10.0, rng, **params)
            ptr = np.zeros(self.g.n, dtype=int)
            while True:
                q = strat.pick()
                if q is None:
                    break
                self.assertFalse(strat.flagged[q], name)
                self.assertLess(ptr[q], sizes[q], name)
                i = ptr[q]
                ptr[q] += 1
                strat.update(q, bool(self.g.pool_succ[q][i]), float(self.g.pool_cost[q][i]))

    def test_adaptive_scalar_and_vector_scores_agree(self):
        sizes = [len(s) for s in self.g.pool_succ]
        rng = np.random.default_rng(5)
        st = Adaptive(sizes, self.g.proj_idx, 4, 10.0, rng, refresh_every=10**9)
        ptr = np.zeros(self.g.n, dtype=int)
        for _ in range(300):
            q = st.pick()
            if q is None:
                break
            i = ptr[q]
            ptr[q] += 1
            st.update(q, bool(self.g.pool_succ[q][i]), float(self.g.pool_cost[q][i]))
        st._refresh()
        vec = st._score_all()
        sc = np.array([st._score_scalar(q) for q in range(self.g.n)])
        fin = np.isfinite(vec)
        self.assertTrue(np.array_equal(fin, np.isfinite(sc)))
        np.testing.assert_allclose(vec[fin], sc[fin], rtol=1e-9)

    def test_paired_orders_identical_across_strategies(self):
        a = run_sim(self.g, *REGISTRY["uniform"], seed=9)
        b = run_sim(self.g, *REGISTRY["uniform"], seed=9)
        np.testing.assert_array_equal(a[0], b[0])

    def test_curves_monotone_and_bounded(self):
        frac = np.linspace(0, 1, 51)
        ev_c, ev_q, _ = run_sim(self.g, *REGISTRY["adaptive"], seed=0)
        cnt = detection_counts(self.g, self.labels, ev_c, ev_q, frac)
        ceil = ceilings(self.g, self.labels)
        self.assertTrue(np.all(np.diff(cnt, axis=2) >= 0))
        self.assertTrue(np.all(cnt[:, :, -1] <= ceil + 1e-9))
        y = cnt[:, FRAGILE, :].sum(0) / ceil[:, FRAGILE].sum()
        self.assertTrue(0 <= audc(y, frac) <= 1)

    def test_skip_drops_variant(self):
        sizes = [3, 3]
        st = REGISTRY["uniform"][0](sizes, np.array([0, 0]), 1, 10.0, np.random.default_rng(0))
        q = st.pick()
        st.skip(q)
        self.assertEqual(st.pool_size[q], 2)


if __name__ == "__main__":
    unittest.main()

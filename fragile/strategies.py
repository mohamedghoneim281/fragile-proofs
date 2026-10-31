"""Test-allocation strategies.

Every strategy answers one question: *which query gets the next test?*  The variant tested is
always the next one in that query's (seeded, random) order, orig first, so strategies differ ONLY
in allocation.  All strategies share the same stopping rules:

  * a query is *flagged* once it has produced both a success and a failure (a witness pair);
  * a flagged query is never tested again;
  * a query whose pool is exhausted is never tested again.

Interface: ``pick() -> query index or None`` then ``update(q, success, cost)`` (always call
update after pick).
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

WARM = 1e9


class Strategy:
    name = "base"

    def __init__(self, pool_size, proj, n_proj, timeout_s, rng, **params):
        self.Q = len(pool_size)
        self.pool_size = np.asarray(pool_size, dtype=np.int64)
        self.proj = np.asarray(proj, dtype=np.int64)
        self.nproj = int(n_proj)
        self.tau = float(timeout_s)
        self.rng = rng
        self.n = np.zeros(self.Q, dtype=np.int64)
        self.s = np.zeros(self.Q, dtype=np.int64)
        self.f = np.zeros(self.Q, dtype=np.int64)
        self.csum = np.zeros(self.Q)
        self.maxsucc = np.zeros(self.Q)
        self.flagged = np.zeros(self.Q, dtype=bool)
        self.params = params

    def eligible(self) -> np.ndarray:
        return (~self.flagged) & (self.n < self.pool_size)

    def _eligible_one(self, q: int) -> bool:
        return (not self.flagged[q]) and self.n[q] < self.pool_size[q]

    def update(self, q: int, success: bool, cost: float) -> None:
        self.n[q] += 1
        self.csum[q] += cost
        if success:
            self.s[q] += 1
            if cost > self.maxsucc[q]:
                self.maxsucc[q] = cost
        else:
            self.f[q] += 1
        if self.s[q] > 0 and self.f[q] > 0:
            self.flagged[q] = True
        self._after_update(q)

    def skip(self, q: int) -> None:
        """The picked variant produced no usable observation (solver error): drop it from the pool."""
        self.pool_size[q] -= 1
        self._after_update(q)

    def _after_update(self, q: int) -> None:  # pragma: no cover
        pass

    def pick(self) -> Optional[int]:  # pragma: no cover
        raise NotImplementedError


class Uniform(Strategy):
    """Breadth first: always test a query that has been tested the fewest times."""

    name = "uniform"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tb = self.rng.permutation(self.Q)
        self.heap = [(0, int(self.tb[q]), q) for q in range(self.Q) if self.pool_size[q] > 0]
        heapq.heapify(self.heap)

    def pick(self):
        return heapq.heappop(self.heap)[2] if self.heap else None

    def _after_update(self, q):
        if self._eligible_one(q):
            heapq.heappush(self.heap, (int(self.n[q]), int(self.tb[q]), q))


class RandomOrder(Strategy):
    """Uniformly random over the remaining (query, variant) pairs."""

    name = "random"

    def pick(self):
        w = np.where(self.eligible(), self.pool_size - self.n, 0)
        cum = np.cumsum(w)
        if self.Q == 0 or cum[-1] <= 0:
            return None
        return int(np.searchsorted(cum, self.rng.random() * cum[-1], side="right"))


class RuntimePriority(Strategy):
    """Simple runtime rule: after ``k0`` warm-up tests, prefer queries that were SOLVED but only
    slowly (largest successful cost / timeout).  Queries that never succeeded come last."""

    name = "runtime"

    def __init__(self, *a, k0: int = 2, **kw):
        super().__init__(*a, **kw)
        self.k0 = int(k0)
        self.jit = self.rng.random(self.Q)
        self.key = np.full(self.Q, -np.inf)
        for q in range(self.Q):
            self.key[q] = self._key_one(q)

    def _key_one(self, q):
        if not self._eligible_one(q):
            return -np.inf
        if self.n[q] < self.k0:
            return WARM + self.jit[q]
        slow = self.maxsucc[q] / self.tau if self.s[q] > 0 else 0.0
        return slow + 1e-6 * self.jit[q]

    def pick(self):
        i = int(np.argmax(self.key))
        return None if self.key[i] == -np.inf else i

    def _after_update(self, q):
        self.key[q] = self._key_one(q)


def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class Adaptive(Strategy):
    """Cost-aware Bayesian allocation with an exploration reserve.

    Model (all of it is a modelling *assumption*, tested by the experiment, not a fact):
      * a query is 'mixed' with probability pi_j (pi_j is estimated per project, shrunk toward the
        global rate, from what has been observed so far) and otherwise 'pure' (always succeeds or
        always fails across harmless variants);
      * a mixed query succeeds with probability theta ~ Beta(a, b) on each variant;
      * after n unanimous tests, P(mixed | data) follows from Bayes' rule; the chance that the next
        test reveals the opposite outcome is P(mixed | data) * E[opposite | mixed, data];
      * a query that was solved but slowly (close to the timeout) gets its odds of being mixed
        multiplied by (1 + gamma * slowest_success / timeout);
      * score = P(next test creates a witness) / expected cost of that test;
      * with probability eps the step is an exploration step (test the least-tested query), so
        queries that look uneventful are still checked.
    No ground-truth labels are used, only the outcomes/costs observed so far.
    """

    name = "adaptive"

    def __init__(
        self,
        *a,
        k0: int = 2,
        eps: float = 0.1,
        pi0: float = 0.05,
        prior_strength: float = 20.0,
        kappa: float = 10.0,
        a_: float = 2.0,
        b_: float = 2.0,
        gamma: float = 2.0,
        c0: float = 1.0,
        pooling: bool = True,
        refresh_every: int = 100,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.k0 = max(1, int(k0))
        self.eps, self.pi0, self.prior_strength, self.kappa = eps, pi0, prior_strength, kappa
        self.a, self.b, self.gamma, self.c0 = a_, b_, gamma, c0
        self.pooling, self.refresh_every = pooling, int(refresh_every)
        self.jit = self.rng.random(self.Q)
        maxn = int(self.pool_size.max()) + 2 if self.Q else 2
        base = _lbeta(self.a, self.b)
        self.Lsucc = np.array([math.exp(_lbeta(self.a + n, self.b) - base) for n in range(maxn)])
        self.Lfail = np.array([math.exp(_lbeta(self.a, self.b + n) - base) for n in range(maxn)])
        self.step = 0
        self.rho_s = 0.5
        self.pi = np.full(self.nproj, self.pi0)
        self.pc = np.full(self.nproj, 0.1 * self.tau)
        self.proj_l = self.proj.tolist()
        self.score = np.full(self.Q, -np.inf)
        self._refresh()

    # ---- scoring ------------------------------------------------------------------
    def _score_scalar(self, q: int) -> float:
        if not self._eligible_one(q):
            return -math.inf
        n = int(self.n[q])
        if n < self.k0:
            return WARM + float(self.jit[q])
        j = self.proj_l[q]
        pi = float(self.pi[j])
        if self.f[q] == 0:
            lm, rho, pnext = float(self.Lsucc[n]), self.rho_s, self.b / (self.a + self.b + n)
        else:
            lm, rho, pnext = float(self.Lfail[n]), 1.0 - self.rho_s, self.a / (self.a + self.b + n)
        odds = pi / (1.0 - pi) * lm / rho
        if self.s[q] > 0:
            odds *= 1.0 + self.gamma * float(self.maxsucc[q]) / self.tau
        pm = odds / (1.0 + odds)
        cost = (float(self.csum[q]) + self.c0 * float(self.pc[j])) / (n + self.c0)
        return pm * pnext / max(cost, 1e-3) * (1.0 + 1e-6 * float(self.jit[q]))

    def _score_all(self) -> np.ndarray:
        el = self.eligible()
        warm = el & (self.n < self.k0)
        nn = self.n
        succ_only = self.f == 0
        lm = np.where(succ_only, self.Lsucc[nn], self.Lfail[nn])
        rho = np.where(succ_only, self.rho_s, 1.0 - self.rho_s)
        pnext = np.where(succ_only, self.b / (self.a + self.b + nn), self.a / (self.a + self.b + nn))
        pi = self.pi[self.proj]
        odds = pi / (1.0 - pi) * lm / rho
        odds = odds * np.where(self.s > 0, 1.0 + self.gamma * self.maxsucc / self.tau, 1.0)
        pm = odds / (1.0 + odds)
        cost = (self.csum + self.c0 * self.pc[self.proj]) / (nn + self.c0)
        sc = pm * pnext / np.maximum(cost, 1e-3) * (1.0 + 1e-6 * self.jit)
        out = np.full(self.Q, -np.inf)
        out[el] = sc[el]
        out[warm] = WARM + self.jit[warm]
        return out

    def _refresh(self) -> None:
        warmed = self.n >= np.minimum(self.k0, self.pool_size)
        m = np.bincount(self.proj[warmed], minlength=self.nproj).astype(float)
        fl = np.bincount(self.proj[self.flagged], minlength=self.nproj).astype(float)
        pi_g = (fl.sum() + self.prior_strength * self.pi0) / (m.sum() + self.prior_strength)
        pi_j = (fl + self.kappa * pi_g) / (m + self.kappa) if self.pooling else np.full(self.nproj, pi_g)
        self.pi = np.clip(pi_j, 1e-3, 0.95)
        tests = np.bincount(self.proj, weights=self.n, minlength=self.nproj)
        cs = np.bincount(self.proj, weights=self.csum, minlength=self.nproj)
        cg = (self.csum.sum() + self.c0 * 0.1 * self.tau) / (self.n.sum() + self.c0)
        self.pc = (cs + self.c0 * cg) / (tests + self.c0)
        us = int((warmed & (self.f == 0) & (self.s > 0)).sum())
        uf = int((warmed & (self.s == 0) & (self.f > 0)).sum())
        self.rho_s = (us + 1.0) / (us + uf + 2.0)
        self.score = self._score_all()

    # ---- interface ----------------------------------------------------------------
    def pick(self):
        if self.eps > 0 and self.rng.random() < self.eps:
            el = self.eligible()
            if not el.any():
                return None
            key = np.where(el, self.n + self.rng.random(self.Q) * 0.5, np.inf)
            return int(np.argmin(key))
        i = int(np.argmax(self.score))
        return None if self.score[i] == -np.inf else i

    def _after_update(self, q):
        self.step += 1
        if self.step % self.refresh_every == 0:
            self._refresh()
        else:
            self.score[q] = self._score_scalar(q)


REGISTRY = {
    "uniform": (Uniform, {}),
    "random": (RandomOrder, {}),
    "runtime": (RuntimePriority, {"k0": 2}),
    "adaptive": (Adaptive, {}),
    "adaptive-nopool": (Adaptive, {"pooling": False}),
    "adaptive-noboost": (Adaptive, {"gamma": 0.0}),
    "adaptive-noexplore": (Adaptive, {"eps": 0.0}),
}

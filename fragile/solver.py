"""Solver backends.

* ``Z3Solver``  - runs the real ``z3`` executable, measures CPU time of the child.
* ``MockSolver`` - SYNTHETIC. Pretends to be a solver using hidden parameters written in a
  ``; mock-params`` comment of the demo queries.  It exists only so the pipeline can be tested
  without a solver.  Results from it say NOTHING about real solvers.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # POSIX only
    import resource  # type: ignore
except ImportError:  # pragma: no cover
    resource = None  # type: ignore

OUTCOMES = ("unsat", "sat", "unknown", "timeout", "memout", "error")
_RESULT_WORDS = ("sat", "unsat", "unknown", "timeout")


@dataclass
class RunResult:
    outcome: str
    wall_s: float
    cpu_s: Optional[float]
    cost_s: float
    detail: str = ""


def _children_cpu() -> Optional[float]:
    if resource is None:
        return None
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def parse_z3_output(stdout: str, returncode: int, wall_s: float, timeout_s: float, stderr: str = "") -> str:
    """Outcome of the FIRST check-sat. Later error lines (e.g. get-model after unsat) are ignored.

    Z3 reports a hit of the ``-memory`` limit as ``(error "out of memory")``; depending on the build and
    the moment it happens this appears on stdout or on stderr, so both are inspected.
    """
    for line in stdout.splitlines():
        s = line.strip()
        if s in _RESULT_WORDS:
            return s
        if "out of memory" in s.lower():
            return "memout"
    if "out of memory" in stderr.lower():
        return "memout"
    if wall_s >= timeout_s - 0.25:
        return "timeout"
    return "error"


def success(outcome: str, mode: str = "unsat") -> bool:
    """What counts as a successful (=proof found) run."""
    if mode == "unsat":
        return outcome == "unsat"
    if mode == "solved":
        return outcome in ("unsat", "sat")
    raise ValueError(mode)


def find_z3(binary: str = "z3") -> Optional[str]:
    """Locate the z3 executable: PATH, then the active Python environment (pip's z3-solver)."""
    hit = shutil.which(binary)
    if hit:
        return hit
    if os.path.isfile(binary):
        return binary
    if binary == "z3":
        import sys

        cands = [os.path.join(sys.prefix, "bin", "z3"), os.path.join(sys.prefix, "Scripts", "z3.exe")]
        try:
            import importlib.util

            spec = importlib.util.find_spec("z3")
            for loc in (spec.submodule_search_locations or []) if spec else []:
                cands += [os.path.join(loc, "bin", "z3"), os.path.join(loc, "bin", "z3.exe")]
        except Exception:
            pass
        for c in cands:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
    return None


class Z3Solver:
    kind = "z3"

    def __init__(self, binary: str = "z3", mem_mb: int = 0, rlimit: int = 0, tmpdir: Optional[str] = None):
        self.binary = find_z3(binary) or binary
        self.mem_mb = int(mem_mb)
        self.rlimit = int(rlimit)
        if tmpdir is None:
            tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        self.tmpdir = tmpdir
        self._version: Optional[str] = None

    @property
    def version(self) -> str:
        if self._version is None:
            for flag in ("--version", "-version"):
                try:
                    out = subprocess.run([self.binary, flag], capture_output=True, text=True, timeout=30).stdout.strip()
                except (OSError, subprocess.TimeoutExpired):
                    out = ""
                if out:
                    self._version = out.splitlines()[0]
                    break
            else:
                self._version = "unknown"
        return self._version

    @property
    def id(self) -> str:
        return f"z3[{self.version}]"

    def command(self, path: str, seed: Optional[int], timeout_s: float) -> List[str]:
        cmd = [self.binary, "-smt2", f"-T:{int(math.ceil(timeout_s))}"]
        if self.mem_mb:
            cmd.append(f"-memory:{self.mem_mb}")
        if self.rlimit:
            cmd.append(f"rlimit={self.rlimit}")
        if seed is not None:
            cmd += [f"smt.random_seed={seed}", f"sat.random_seed={seed}"]
        cmd.append(path)
        return cmd

    def run(self, text: str, seed: Optional[int], timeout_s: float, repeat: int = 0) -> RunResult:
        with tempfile.NamedTemporaryFile("w", suffix=".smt2", dir=self.tmpdir, delete=False) as fh:
            fh.write(text)
            path = fh.name
        cmd = self.command(path, seed, timeout_s)
        cpu0 = _children_cpu()
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 5)
            stdout, stderr, rc, killed = p.stdout, p.stderr, p.returncode, False
        except subprocess.TimeoutExpired:
            stdout, stderr, rc, killed = "", "", -9, True
        finally:
            wall = time.perf_counter() - t0
            try:
                os.unlink(path)
            except OSError:
                pass
        cpu1 = _children_cpu()
        cpu = None if cpu0 is None or cpu1 is None else max(0.0, cpu1 - cpu0)
        outcome = "timeout" if killed else parse_z3_output(stdout, rc, wall, timeout_s, stderr)
        # Timeouts are charged the full timeout so budgets are comparable across machines/load.
        cost = timeout_s if outcome == "timeout" else max(1e-3, cpu if cpu is not None else wall)
        detail = (stdout[:150] + " || " + stderr[:150]).replace("\n", " | ")
        return RunResult(outcome, wall, cpu, cost, detail=detail)


_MOCK_RE = re.compile(r"mock-params\s+mu=([-+\d.eE]+)\s+sigma=([-+\d.eE]+)\s+p_unknown=([-+\d.eE]+)")
_NORM = statistics.NormalDist()


def _unif(*parts) -> float:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    u = (int.from_bytes(h, "big") + 0.5) / 2**64
    return min(max(u, 1e-9), 1 - 1e-9)


class MockSolver:
    """SYNTHETIC solver: runtime is lognormal with hidden per-query parameters. Pipeline tests only."""

    kind = "mock"
    id = "mock[SYNTHETIC]"
    version = "synthetic"

    def run(self, text: str, seed: Optional[int], timeout_s: float, repeat: int = 0) -> RunResult:
        m = _MOCK_RE.search(text[:4000])
        if not m:
            raise ValueError("mock solver needs a '; mock-params mu=.. sigma=.. p_unknown=..' header")
        mu, sigma, p_unknown = (float(x) for x in m.groups())
        th = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
        eps = _NORM.inv_cdf(_unif(th, seed, "eps"))
        noise = _NORM.inv_cdf(_unif(th, seed, repeat, "noise"))
        runtime = math.exp(mu + sigma * eps + 0.03 * noise)
        if runtime > timeout_s:
            return RunResult("timeout", timeout_s, timeout_s, timeout_s, "mock")
        if _unif(th, seed, "unk") < p_unknown:
            c = max(1e-3, 0.5 * runtime)
            return RunResult("unknown", c, c, c, "mock")
        c = max(1e-3, runtime)
        return RunResult("unsat", c, c, c, "mock")


def build_solver(spec: Dict):
    kind = spec.get("kind", "z3")
    if kind == "mock":
        return MockSolver()
    if kind == "z3":
        return Z3Solver(spec.get("binary", "z3"), spec.get("mem_mb", 0), spec.get("rlimit", 0))
    raise ValueError(f"unknown solver kind {kind!r}")

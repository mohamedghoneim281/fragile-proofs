"""`python -m fragile check`: verify the machine is ready BEFORE a long run."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

from .smt2 import Script
from .solver import Z3Solver, find_z3
from .variants import planned_variants, render_variant

TINY = """(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-const c Int)
(assert (forall ((x Int)) (>= (f x) x)))
(assert (< (f c) c))
(assert (> c 0))
(check-sat)
(get-model)
"""


def check_command(args) -> int:
    ok_all = True

    def line(ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(("[ OK ] " if ok else "[FAIL] ") + msg)

    line(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]} (need 3.9+)")
    for mod in ("numpy", "matplotlib"):
        try:
            __import__(mod)
            line(True, f"{mod} installed")
        except ImportError:
            line(False, f"{mod} missing -> pip install -r requirements.txt")

    z3 = Z3Solver(args.solver_bin)
    found = find_z3(args.solver_bin)
    line(found, f"solver executable found: {found}" if found else
         f"solver '{args.solver_bin}' NOT found -> run:  pip install -r requirements.txt   (inside the virtual environment); "
         "if that was already done, run:  sudo apt install -y z3   and then this check again")
    if not found:
        print("\nFix the FAIL lines above, then run this check again.")
        return 1
    print(f"       version: {z3.version}")

    def run(label, solver, seed=None, timeout=10, text=TINY):
        r = solver.run(text, seed, timeout)
        line(r.outcome == "unsat", f"{label}: outcome={r.outcome} ({r.wall_s:.2f}s wall, cpu={r.cpu_s})")
        if r.outcome != "unsat":
            print("       solver said:", r.detail)
        return r

    run("basic run (with a get-model after unsat, which must be ignored)", z3)
    run("with random seed flags (smt.random_seed / sat.random_seed)", z3, seed=12345)
    if args.mem_mb:
        run(f"with memory limit -memory:{args.mem_mb}", Z3Solver(args.solver_bin, mem_mb=args.mem_mb))
    if args.rlimit:
        run(f"with resource limit rlimit={args.rlimit}", Z3Solver(args.solver_bin, rlimit=args.rlimit))

    s = Script(TINY)
    vs = planned_variants(s, ("rename", "shuffle", "reseed", "mix"), 2)
    bad = 0
    for vid in vs:
        text, seed = render_variant(s, "tiny", vid, 0)
        bad += z3.run(text, seed, 10).outcome != "unsat"
    line(bad == 0, f"all {len(vs)} variants of a tiny query give the same answer as the original")

    cpu = z3.run(TINY, None, 10).cpu_s
    line(cpu is not None, "CPU time of the solver process can be measured" if cpu is not None else
         "CPU time unavailable on this OS -> wall time is used (Linux/macOS recommended)")
    try:
        load = os.getloadavg()[0]
        line(load < max(1.0, (os.cpu_count() or 1) / 2), f"machine load {load:.1f} on {os.cpu_count()} cores (close other programs for cleaner timing)")
    except OSError:
        pass
    print("\nAll good - you can start collecting." if ok_all else "\nSome checks failed - fix them before a long run.")
    return 0 if ok_all else 1

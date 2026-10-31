"""Command line: python -m fragile <command> --help"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from . import __version__


def _solver_args(p, with_limits=True):
    p.add_argument("--solver", default="z3", choices=["z3", "mock"], help="z3 = real solver (default); mock = SYNTHETIC test backend")
    p.add_argument("--solver-bin", default="z3", help="path or name of the z3 executable (default: z3 on PATH)")
    if with_limits:
        p.add_argument("--mem-mb", type=int, default=4096, help="memory limit per solver run (0 = none)")
        p.add_argument("--rlimit", type=int, default=0, help="optional Z3 resource limit (0 = off); more reproducible than time")


def _query_args(p):
    p.add_argument("--queries", required=True, help="folder containing .smt2 files")
    p.add_argument("--project-depth", type=int, default=1, help="how many folder levels below --queries name a project (default 1)")
    p.add_argument("--manifest", help="optional CSV with columns path,project (overrides folder-based projects)")
    p.add_argument("--per-project", type=int, default=0, help="max queries per project (0 = all)")
    p.add_argument("--max-queries", type=int, default=0, help="max queries in total (0 = all)")
    p.add_argument("--max-mb", type=float, default=20.0, help="skip queries larger than this")
    p.add_argument("--families", default="rename,shuffle,reseed", help="comma list from rename,shuffle,reseed,mix")
    p.add_argument("-k", type=int, default=10, help="variants per family (use an even number; half go to the reference set)")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds per solver run")
    p.add_argument("--seed", type=int, default=0, help="base seed (selection + variants)")
    p.add_argument("--success", default="unsat", choices=["unsat", "solved"], help="what counts as success")
    p.add_argument("--force", action="store_true", help="allow mixing settings in one --out folder (not recommended)")


def build_parser():
    ap = argparse.ArgumentParser(prog="fragile", description="Find fragile SMT proofs on a small compute budget.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="verify Python libs, Z3 and the exact flags this tool uses")
    p.add_argument("--solver-bin", default="z3")
    p.add_argument("--mem-mb", type=int, default=4096)
    p.add_argument("--rlimit", type=int, default=0)

    p = sub.add_parser("demo", help="end-to-end test on SYNTHETIC data with the mock solver (no Z3 needed)")
    p.add_argument("--out", default="demo_out")
    p.add_argument("--seeds", type=int, default=10)

    p = sub.add_parser("scan", help="list projects/queries found in a folder and check they are usable")
    p.add_argument("queries")
    p.add_argument("--project-depth", type=int, default=1)
    p.add_argument("--manifest")
    p.add_argument("--sample", type=int, default=40, help="queries per project to parse-check")

    p = sub.add_parser("prepare-mariposa", help="build a manifest (project + curated group) from a mariposa-data 'benchmarks' folder")
    p.add_argument("source", help="path to mariposa-data/benchmarks")
    p.add_argument("--out", default="data_mariposa")

    p = sub.add_parser("collect", help="measure the (query x variant) grid (resumable)")
    _query_args(p)
    _solver_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--pin", action="store_true", help="pin each worker to its own CPU core (Linux)")
    p.add_argument("--control-repeats", type=int, default=3, help="runs of the unchanged original (>=2 gives noise controls)")
    p.add_argument("--max-hours", type=float, default=0, help="stop after this many wall hours (resume later); 0 = no limit")
    p.add_argument("--retry-errors", action="store_true", help="measure again every run whose latest record is a solver 'error'")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("status", help="progress, outcome counts, errors, load, noise controls")
    p.add_argument("out")

    p = sub.add_parser("evaluate", help="replay strategies on collected data; writes results/ with figures and summary.md")
    p.add_argument("out")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.1, help="fragile = reference success rate in [tau, 1-tau]")
    p.add_argument("--label-mode", default="independent", choices=["independent", "full"])
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--tune-grid", default="small", choices=["small", "full"])
    p.add_argument("--tune-seeds", type=int, default=3)
    p.add_argument("--ablations", action="store_true", help="also run adaptive without pooling / boost / exploration")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="fewer seeds, no tuning (fast sanity pass)")

    p = sub.add_parser("hunt", help="LIVE: run a strategy on a real solver with a CPU budget")
    _query_args(p)
    _solver_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--strategy", default="adaptive", choices=["uniform", "random", "runtime", "adaptive"])
    p.add_argument("--budget-hours", type=float, default=1.0, help="solver CPU hours to spend")

    p = sub.add_parser("confirm", help="re-run witness pairs to separate input sensitivity from timing noise; writes evidence/")
    p.add_argument("out")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max", type=int, default=30, help="max witnesses to confirm (cheapest first)")
    p.add_argument("--only-hunt", action="store_true", help="only witnesses found by `hunt`")
    p.add_argument("--solver", default=None, choices=[None, "z3", "mock"])
    p.add_argument("--solver-bin", default=None)
    return ap


def cmd_scan(a):
    from .collect import discover, inspect_query

    infos = discover(a.queries, a.project_depth, a.manifest)
    by = {}
    for i in infos:
        by.setdefault(i.project, []).append(i)
    print(f"{len(infos)} .smt2 files in {len(by)} projects under {a.queries}\n")
    print(f"{'project':40s} {'queries':>8s} {'median MB':>10s} {'usable*':>9s}")
    rng = random.Random(0)
    for proj in sorted(by):
        lst = by[proj]
        sample = rng.sample(lst, min(a.sample, len(lst)))
        ok = sum(inspect_query(i, ("rename", "shuffle", "reseed"), 2, 10**9)["ok"] for i in sample)
        sizes = sorted(i.nbytes for i in lst)
        print(f"{proj[:40]:40s} {len(lst):8d} {sizes[len(sizes) // 2] / 1e6:10.2f} {ok:>4d}/{len(sample):<4d}")
    print("\n* usable = parses and has exactly one check-sat, among a sample of the project's queries")


def cmd_status(a):
    from .report import status_text

    print(status_text(a.out))


def cmd_evaluate(a):
    from .evaluate import DEFAULT_STRATEGIES, evaluate
    from .grid import load_grid
    from .report import make_figures, noise_controls, save_json, summary_markdown
    from .store import Store

    grid = load_grid(a.out)
    rep = grid.report
    print(f"loaded {grid.n} complete queries in {len(grid.proj_names)} projects "
          f"(incomplete {rep['incomplete']}, original errors {rep['orig_error']}, variant errors skipped {rep['variant_errors']})")
    if rep["conflicts"]:
        print(f"NOTE: {len(rep['conflicts'])} queries returned both sat and unsat on equivalent variants - investigate!")
    seeds, tune = (4, False) if a.quick else (a.seeds, not a.no_tune)
    strategies = list(DEFAULT_STRATEGIES)
    if a.ablations:
        strategies += ["adaptive-nopool", "adaptive-noboost", "adaptive-noexplore"]
    lab_all = grid.labels(a.tau, a.label_mode)
    crosstab = {}
    if any(grid.groups):
        names = {0: "pure", 1: "borderline", 2: "fragile"}
        for g, l in zip(grid.groups, lab_all):
            crosstab.setdefault(g, {"pure": 0, "borderline": 0, "fragile": 0})[names[int(l)]] += 1
    res = evaluate(grid, seeds=seeds, folds=a.folds, tau=a.tau, label_mode=a.label_mode, strategies=strategies,
                   tune=tune, tune_seeds=a.tune_seeds, tune_grid=a.tune_grid, boot=a.boot)
    res["group_crosstab"] = crosstab
    outdir = Path(a.out) / "results"
    outdir.mkdir(exist_ok=True)
    synthetic = bool(grid.meta.get("synthetic"))
    figs = make_figures(res, outdir, synthetic)
    noise = noise_controls(Store(a.out).read(), grid.meta.get("success_mode", "unsat"))
    save_json(outdir / "eval.json", {**res, "grid_report": rep, "noise": noise})
    md = summary_markdown(res, rep, grid.meta, noise, figs)
    (outdir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"\nwritten to {outdir}/ (summary.md, eval.json, fig*.png)")


def cmd_demo(a):
    from .collect import collect_command
    from .demo import make_demo

    print("*** SYNTHETIC demo: tests the pipeline only; the numbers mean nothing about real solvers ***\n")
    qdir = Path(a.out + "_queries")
    n = make_demo(qdir, n_projects=8, per_project=15, timeout_s=10.0)
    print(f"generated {n} synthetic queries in {qdir}")
    ns = argparse.Namespace(
        queries=str(qdir), project_depth=1, manifest=None, per_project=0, max_queries=0, max_mb=20.0,
        families="rename,shuffle,reseed", k=10, timeout=10.0, seed=0, success="unsat", force=True,
        solver="mock", solver_bin="z3", mem_mb=0, rlimit=0, out=a.out, workers=1, pin=False,
        control_repeats=3, max_hours=0, dry_run=False)
    collect_command(ns)
    ev = argparse.Namespace(out=a.out, seeds=a.seeds, folds=3, tau=0.1, label_mode="independent", no_tune=False,
                            tune_grid="small", tune_seeds=2, ablations=False, boot=500, quick=False)
    cmd_evaluate(ev)


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "check":
        from .check import check_command

        sys.exit(check_command(a))
    elif a.cmd == "demo":
        cmd_demo(a)
    elif a.cmd == "scan":
        cmd_scan(a)
    elif a.cmd == "prepare-mariposa":
        from .mariposa import prepare_mariposa

        prepare_mariposa(a.source, a.out)
    elif a.cmd == "collect":
        from .collect import collect_command

        collect_command(a)
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "evaluate":
        cmd_evaluate(a)
    elif a.cmd == "hunt":
        from .live import hunt_command

        hunt_command(a)
    elif a.cmd == "confirm":
        from .live import confirm_command

        confirm_command(a)

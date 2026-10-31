"""Find queries, choose a pilot subset, and measure the (query x variant) grid."""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .smt2 import Script
from .solver import build_solver
from .store import Store, host_info
from .variants import ORIG, planned_variants, render_variant


@dataclass
class QueryInfo:
    qid: str
    path: str
    project: str
    nbytes: int
    group: str = ""


# ------------------------------------------------------------------ discovery ---------
def discover(root, project_depth: int = 1, manifest: Optional[str] = None) -> List[QueryInfo]:
    """All ``*.smt2`` files below ``root``.  The project (family) of a query is the first
    ``project_depth`` folder names below ``root`` (or the ``project`` column of a CSV manifest)."""
    root = Path(root).resolve()
    infos: List[QueryInfo] = []
    if manifest:
        with open(manifest, newline="") as fh:
            for row in csv.DictReader(fh):
                p = Path(row["path"])
                p = p if p.is_absolute() else root / p
                try:
                    qid = p.resolve().relative_to(root).as_posix()
                except ValueError:
                    qid = p.as_posix()
                infos.append(QueryInfo(qid, str(p), row["project"], p.stat().st_size, row.get("group", "") or ""))
        return infos
    for p in sorted(root.rglob("*.smt2")):
        parts = p.relative_to(root).parts
        if project_depth <= 0:
            project = "default"
        elif len(parts) > project_depth:
            project = "/".join(parts[:project_depth])
        else:
            project = "/".join(parts[:-1]) or "default"
        infos.append(QueryInfo(p.relative_to(root).as_posix(), str(p), project, p.stat().st_size))
    return infos


def inspect_query(info: QueryInfo, families: Sequence[str], k: int, max_bytes: int) -> Dict:
    """Parse the query and decide whether (and how) it can be used."""
    out = {"ok": False, "reason": ""}
    if info.nbytes > max_bytes:
        out["reason"] = f"larger than --max-mb ({info.nbytes / 1e6:.1f} MB)"
        return out
    try:
        with open(info.path, encoding="utf-8", errors="replace") as fh:
            script = Script(fh.read())
    except (ValueError, OSError) as e:
        out["reason"] = f"unreadable/unbalanced: {e}"
        return out
    n_cs = script.count("check-sat")
    if n_cs != 1:
        out["reason"] = f"{n_cs} check-sat commands (need exactly 1)"
        return out
    variants = planned_variants(script, families, k)
    out.update(ok=True, variants=variants, declared=len(script.declared), can_shuffle=script.can_shuffle)
    return out


def select_queries(infos: List[QueryInfo], per_project: int, max_queries: int, seed: int) -> List[QueryInfo]:
    """Random subset, balanced over projects, returned in project-interleaved random order
    (so a run stopped half-way still contains complete, balanced queries)."""
    rng = random.Random(seed)
    by_proj: Dict[str, List[QueryInfo]] = OrderedDict()
    for i in infos:
        by_proj.setdefault(i.project, []).append(i)
    for lst in by_proj.values():
        rng.shuffle(lst)
        if per_project > 0:
            del lst[per_project:]
    projs = list(by_proj)
    rng.shuffle(projs)
    out: List[QueryInfo] = []
    i = 0
    while any(by_proj[p] for p in projs):
        for p in projs:
            if by_proj[p]:
                out.append(by_proj[p].pop())
        i += 1
    if max_queries > 0:
        out = out[:max_queries]
    return out


def prepare_manifest(store: Store, infos: List[QueryInfo], families, k, max_bytes, log=print) -> List[Dict]:
    manifest, skipped = [], []
    for info in infos:
        res = inspect_query(info, families, k, max_bytes)
        if res["ok"]:
            manifest.append({**asdict(info), "variants": res["variants"]})
        else:
            skipped.append({"qid": info.qid, "reason": res["reason"]})
    old = store.load_json("queries.json", [])
    known = {m["qid"] for m in old}
    merged = old + [m for m in manifest if m["qid"] not in known]
    store.save_json("queries.json", merged)
    store.save_json("skipped.json", skipped)
    if skipped:
        log(f"skipped {len(skipped)} unusable queries (see skipped.json)")
    return manifest


# ------------------------------------------------------------------ workers -----------
_W: Dict = {}


def _init_worker(cfg: Dict, counter, pin: bool) -> None:
    _W["solver"] = build_solver(cfg["solver"])
    _W["cfg"] = cfg
    _W["cache"] = OrderedDict()
    if pin and hasattr(os, "sched_setaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        with counter.get_lock():
            idx = counter.value
            counter.value += 1
        try:
            os.sched_setaffinity(0, {cores[idx % len(cores)]})
        except OSError:
            pass


def _get_script(path: str) -> Script:
    cache = _W["cache"]
    if path in cache:
        cache.move_to_end(path)
        return cache[path]
    with open(path, encoding="utf-8", errors="replace") as fh:
        s = Script(fh.read())
    cache[path] = s
    while len(cache) > 2:
        cache.popitem(last=False)
    return s


def _load1() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return -1.0


def run_task(task: Tuple[str, str, str, int]) -> Dict:
    qid, path, vid, repeat = task
    cfg = _W["cfg"]
    script = _get_script(path)
    load = _load1()
    text, seed = render_variant(script, qid, vid, cfg["base_seed"])
    res = _W["solver"].run(text, seed, cfg["timeout_s"], repeat)
    return {
        "query": qid,
        "variant": vid,
        "repeat": repeat,
        "outcome": res.outcome,
        "wall_s": round(res.wall_s, 4),
        "cpu_s": None if res.cpu_s is None else round(res.cpu_s, 4),
        "cost_s": round(res.cost_s, 4),
        "load1": round(load, 2),
        "ts": int(time.time()),
    }


def build_tasks(manifest: List[Dict], control_repeats: int, done) -> List[Tuple[str, str, str, int]]:
    tasks = []
    for m in manifest:
        for vid in m["variants"]:
            if (m["qid"], vid, 0) not in done:
                tasks.append((m["qid"], m["path"], vid, 0))
        for r in range(1, control_repeats):
            if (m["qid"], ORIG, r) not in done:
                tasks.append((m["qid"], m["path"], ORIG, r))
    return tasks


def run_tasks(tasks, cfg, store: Store, workers: int, pin: bool, max_hours: float, log=print) -> Dict:
    t0 = time.time()
    n = len(tasks)
    stats = {"done": 0, "cost": 0.0, "outcomes": {}}
    if n == 0:
        return stats
    deadline = t0 + max_hours * 3600 if max_hours > 0 else None
    counter = mp.Value("i", 0)

    def handle(rec):
        store.append(rec)
        stats["done"] += 1
        stats["cost"] += rec["cost_s"]
        stats["outcomes"][rec["outcome"]] = stats["outcomes"].get(rec["outcome"], 0) + 1
        d = stats["done"]
        if d % max(1, min(200, n // 50 or 1)) == 0 or d == n:
            el = time.time() - t0
            eta = el / d * (n - d)
            log(
                f"  {d}/{n} runs  elapsed {el / 60:5.1f} min  eta {eta / 60:6.1f} min  "
                f"solver CPU so far {stats['cost'] / 3600:6.2f} h  load {rec['load1']}"
            )

    try:
        if workers <= 1:
            _init_worker(cfg, counter, False)
            for t in tasks:
                handle(run_task(t))
                if deadline and time.time() > deadline:
                    log("  --max-hours reached; stopping (re-run the same command to resume)")
                    break
        else:
            with mp.Pool(workers, initializer=_init_worker, initargs=(cfg, counter, pin)) as pool:
                try:
                    for rec in pool.imap_unordered(run_task, tasks, chunksize=1):
                        handle(rec)
                        if deadline and time.time() > deadline:
                            log("  --max-hours reached; stopping (re-run the same command to resume)")
                            pool.terminate()
                            break
                except KeyboardInterrupt:
                    log("  interrupted: progress is saved; re-run the same command to resume")
                    pool.terminate()
    except KeyboardInterrupt:
        log("  interrupted: progress is saved; re-run the same command to resume")
    return stats


# ------------------------------------------------------------------ command -----------
def collect_command(args) -> None:
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    store = Store(args.out)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    solver_spec = {"kind": args.solver, "binary": args.solver_bin, "mem_mb": args.mem_mb, "rlimit": args.rlimit}
    solver = build_solver(solver_spec)

    infos = discover(args.queries, args.project_depth, args.manifest)
    if not infos:
        raise SystemExit(f"No .smt2 files found below {args.queries}")
    chosen = select_queries(infos, args.per_project, args.max_queries, args.seed)
    log(f"found {len(infos)} queries in {len({i.project for i in infos})} projects; selected {len(chosen)}")

    meta = store.check_and_update_meta(
        {
            "solver_id": solver.id,
            "solver_kind": solver_spec["kind"],
            "solver_binary": getattr(solver, "binary", None),
            "timeout_s": args.timeout,
            "base_seed": args.seed,
            "success_mode": args.success,
            "rlimit": args.rlimit,
            "mem_mb": args.mem_mb,
            "families": families,
            "variants_per_family": args.k,
            "control_repeats": args.control_repeats,
            "synthetic": solver_spec["kind"] == "mock",
            "host": host_info(),
            "command": " ".join(sys.argv),
        },
        force=args.force,
    )
    manifest = prepare_manifest(store, chosen, families, args.k, int(args.max_mb * 1e6), log)
    # also resume queries chosen earlier in this folder
    all_manifest = store.load_json("queries.json", [])
    chosen_ids = {m["qid"] for m in manifest}
    manifest = [m for m in all_manifest if m["qid"] in chosen_ids] or manifest

    done = store.done_keys(retry_errors=getattr(args, 'retry_errors', False))
    tasks = build_tasks(manifest, args.control_repeats, done)
    n_variants = sum(len(m["variants"]) for m in manifest)
    log(f"{len(manifest)} usable queries, {n_variants} planned variant runs, {len(tasks)} still to do")
    log(f"worst case if every run timed out: {len(tasks) * args.timeout / 3600:.1f} CPU hours "
        f"(typical cost is far lower; the progress line shows the real running total)")
    if solver_spec["kind"] == "mock":
        log("*** SYNTHETIC mock solver: results are for testing the pipeline only ***")
    if args.dry_run:
        log("dry run: nothing executed")
        return
    cfg = {"solver": solver_spec, "timeout_s": args.timeout, "base_seed": args.seed}
    stats = run_tasks(tasks, cfg, store, args.workers, args.pin, args.max_hours, log)
    store.close()
    log(f"finished {stats['done']} runs; outcomes: {stats['outcomes']}")
    if stats["outcomes"].get("error", 0):
        log("WARNING: some runs ended in 'error'. Run `python -m fragile status` to inspect them.")

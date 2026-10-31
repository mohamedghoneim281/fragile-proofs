"""Live use of the strategies on a real solver (`hunt`) and re-run confirmation (`confirm`)."""
from __future__ import annotations

import json
import random
import re
import shlex
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .collect import discover, prepare_manifest, select_queries
from .report import save_json
from .smt2 import Script
from .solver import build_solver, success
from .store import Store, host_info
from .strategies import REGISTRY
from .variants import ORIG, render_variant


def _solver_spec(args, meta: Optional[Dict] = None) -> Dict:
    meta = meta or {}
    return {
        "kind": args.solver or meta.get("solver_kind", "z3"),
        "binary": args.solver_bin or meta.get("solver_binary") or "z3",
        "mem_mb": meta.get("mem_mb", getattr(args, "mem_mb", 0)),
        "rlimit": meta.get("rlimit", getattr(args, "rlimit", 0)),
    }


# ------------------------------------------------------------------ hunt -----------------
def hunt_command(args) -> None:
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    store = Store(args.out)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    spec = {"kind": args.solver, "binary": args.solver_bin, "mem_mb": args.mem_mb, "rlimit": args.rlimit}
    solver = build_solver(spec)
    store.check_and_update_meta({
        "solver_id": solver.id, "solver_kind": spec["kind"], "solver_binary": getattr(solver, "binary", None),
        "timeout_s": args.timeout, "base_seed": args.seed, "success_mode": args.success, "rlimit": args.rlimit,
        "mem_mb": args.mem_mb, "families": families, "variants_per_family": args.k, "mode": f"hunt:{args.strategy}",
        "synthetic": spec["kind"] == "mock", "host": host_info(),
    }, force=args.force)
    infos = discover(args.queries, args.project_depth, args.manifest)
    chosen = select_queries(infos, args.per_project, args.max_queries, args.seed)
    manifest = prepare_manifest(store, chosen, families, args.k, int(args.max_mb * 1e6), log)
    if not manifest:
        raise SystemExit("no usable queries")
    names = sorted({m["project"] for m in manifest})
    proj = np.array([names.index(m["project"]) for m in manifest])
    rng = random.Random(args.seed)
    order = []
    for m in manifest:
        rest = m["variants"][1:]
        rng.shuffle(rest)
        order.append([m["variants"][0]] + rest)
    cls, params = REGISTRY[args.strategy]
    strat = cls([len(o) for o in order], proj, len(names), args.timeout, np.random.default_rng(args.seed), **params)
    cache = {(r["query"], r["variant"]): r for r in store.read() if r["repeat"] == 0}
    scripts: Dict[int, Script] = {}
    ptr = [0] * len(manifest)
    seen: List[List] = [[] for _ in manifest]
    budget = args.budget_hours * 3600.0
    spent, t0, steps = 0.0, time.time(), 0
    log(f"hunting with '{args.strategy}' on {len(manifest)} queries, budget {args.budget_hours} CPU h")
    try:
        while spent < budget:
            q = strat.pick()
            if q is None:
                break
            vid = order[q][ptr[q]]
            ptr[q] += 1
            m = manifest[q]
            key = (m["qid"], vid)
            if key not in cache:
                if q not in scripts:
                    scripts.clear()
                    scripts[q] = Script(Path(m["path"]).read_text(errors="replace"))
                text, seed = render_variant(scripts[q], m["qid"], vid, args.seed)
                res = solver.run(text, seed, args.timeout)
                rec = {"query": m["qid"], "variant": vid, "repeat": 0, "outcome": res.outcome, "wall_s": round(res.wall_s, 4),
                       "cpu_s": res.cpu_s, "cost_s": round(res.cost_s, 4), "load1": -1, "ts": int(time.time())}
                store.append(rec)
                cache[key] = rec
            rec = cache[key]
            if rec["outcome"] == "error":
                strat.skip(q)  # an error is not a failure: drop that variant, never report it as a witness
                seen[q].append((vid, None))
                continue
            ok = success(rec["outcome"], args.success)
            spent += rec["cost_s"]
            seen[q].append((vid, ok))
            strat.update(q, ok, rec["cost_s"])
            steps += 1
            if steps % 100 == 0:
                log(f"  {steps} tests, {spent / 3600:.2f}/{args.budget_hours} CPU h, flagged {int(strat.flagged.sum())}")
    except KeyboardInterrupt:
        log("interrupted; saving what was found")
    store.close()
    found = []
    for q, m in enumerate(manifest):
        good = [v for v, o in seen[q] if o is True]
        bad = [v for v, o in seen[q] if o is False]
        if good and bad:
            found.append({"query": m["qid"], "project": m["project"], "success_variant": good[0], "failure_variant": bad[0]})
    save_json(Path(args.out) / "hunt.json", {"strategy": args.strategy, "spent_cpu_s": spent, "tests": steps, "flagged": found})
    log(f"done: {steps} tests, {spent / 3600:.2f} CPU h, {len(found)} candidate fragile queries -> {args.out}/hunt.json")
    log("next: python -m fragile confirm " + str(args.out))


# ------------------------------------------------------------------ confirm --------------
def find_witnesses(store: Store, mode: str) -> List[Dict]:
    obs: Dict[str, Dict[str, Dict]] = {}
    for r in store.read():
        if r["repeat"] == 0 and r["outcome"] != "error":
            obs.setdefault(r["query"], {})[r["variant"]] = r
    out = []
    for qid, vs in obs.items():
        good = [(v, r) for v, r in vs.items() if success(r["outcome"], mode)]
        bad = [(v, r) for v, r in vs.items() if not success(r["outcome"], mode)]
        if not good or not bad:
            continue
        # prefer the story "the original is fine, a harmless variant is not" (or the reverse)
        a = next(((v, r) for v, r in good if v == ORIG), None) or next(((v, r) for v, r in bad if v == ORIG), None)
        if a is None:
            a = good[0]
        a_ok = success(a[1]["outcome"], mode)
        pool = bad if a_ok else good
        b = min(pool, key=lambda x: x[1]["cost_s"])
        out.append({"query": qid, "a": a[0], "a_rec": a[1], "b": b[0], "b_rec": b[1],
                    "cost": a[1]["cost_s"] + b[1]["cost_s"]})
    out.sort(key=lambda w: w["cost"])
    return out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]


def confirm_command(args) -> None:
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    store = Store(args.out)
    meta = store.load_json("meta.json", {})
    mode = meta.get("success_mode", "unsat")
    timeout = float(meta["timeout_s"])
    base_seed = int(meta["base_seed"])
    solver = build_solver(_solver_spec(args, meta))
    paths = {m["qid"]: m["path"] for m in store.load_json("queries.json", [])}
    wit = find_witnesses(store, mode)
    if args.only_hunt and (Path(args.out) / "hunt.json").exists():
        keep = {f["query"] for f in json.loads((Path(args.out) / "hunt.json").read_text())["flagged"]}
        wit = [w for w in wit if w["query"] in keep]
    wit = wit[: args.max]
    log(f"{len(wit)} candidate witnesses; re-running each side {args.repeats} times")
    results = []
    ev_root = Path(args.out) / "evidence"
    for w in wit:
        script = Script(Path(paths[w["query"]]).read_text(errors="replace"))
        sides = {}
        for key in ("a", "b"):
            vid = w[key]
            text, seed = render_variant(script, w["query"], vid, base_seed)
            recorded = success(w[key + "_rec"]["outcome"], mode)
            same, outs = 0, []
            for r in range(args.repeats):
                res = solver.run(text, seed, timeout, repeat=r + 1)
                outs.append(res.outcome)
                same += success(res.outcome, mode) == recorded
            sides[key] = {"variant": vid, "recorded": w[key + "_rec"]["outcome"], "reruns": outs,
                          "reproduced_frac": same / args.repeats, "text": text, "seed": seed}
        confirmed = all(s["reproduced_frac"] >= 0.8 for s in sides.values())
        results.append({"query": w["query"], "confirmed": confirmed,
                        **{k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in sides.items()}})
        if confirmed:
            d = ev_root / _safe(w["query"])
            d.mkdir(parents=True, exist_ok=True)
            lines = ["#!/usr/bin/env bash", "# Re-runs two meaning-preserving variants of one query. Set Z3=/path/to/z3 to choose the binary.",
                     'Z3=${Z3:-z3}']
            for key in ("a", "b"):
                s = sides[key]
                fn = f"{key.upper()}_{_safe(s['variant'])}.smt2"
                (d / fn).write_text(s["text"])
                if solver.kind == "z3":
                    cmd = solver.command(fn, s["seed"], timeout)
                    cmd = ["$Z3"] + [shlex.quote(c) for c in cmd[1:]]
                    lines.append(f'echo "== {key.upper()}: variant {s["variant"]}, recorded {s["recorded"]}"')
                    lines.append("time " + " ".join(cmd))
            (d / "reproduce.sh").write_text("\n".join(lines) + "\n")
            (d / "README.txt").write_text(
                f"query: {w['query']}\nsolver: {meta.get('solver_id')}\ntimeout: {timeout}s\n"
                f"A = {sides['a']['variant']}: recorded {sides['a']['recorded']}, reruns {sides['a']['reruns']}\n"
                f"B = {sides['b']['variant']}: recorded {sides['b']['recorded']}, reruns {sides['b']['reruns']}\n"
                "Both files are meaning-preserving variants of the same query (symbol renaming / assertion order / solver seed).\n")
        log(f"  {w['query']}: {'CONFIRMED' if confirmed else 'not reproduced'} "
            f"(A {sides['a']['reproduced_frac']:.0%}, B {sides['b']['reproduced_frac']:.0%})")
    save_json(Path(args.out) / "confirmations.json", results)
    n_ok = sum(r["confirmed"] for r in results)
    log(f"confirmed {n_ok}/{len(results)}; bundles in {ev_root}")

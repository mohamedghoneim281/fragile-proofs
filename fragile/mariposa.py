"""`prepare-mariposa`: turn a checkout of the Mariposa benchmark into a manifest CSV.

The folder layout of the benchmark was not visible when this was written, so the project
(e.g. a Komodo / VeriBetrKV / DICE / vWasm variant) and the curated group (unstable_core,
unstable_ext, stable_core, stable_ext) are *inferred from the path* wherever they appear
(folder names or file name).  The inference is printed as a table so it can be checked.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

GROUP_RE = re.compile(r"(un)?stable[_\- ]?(core|ext\w*)", re.I)
PROJECT_RE = re.compile(r"(komodo\w*|veribetr\w*|fvbkv\w*|lvbkv\w*|dice\w*|vwasm\w*)", re.I)
GROUP_ORDER = {"unstable_core": 0, "stable_core": 1, "unstable_ext": 2, "stable_ext": 3}


def _norm_group(s: str) -> str:
    m = GROUP_RE.search(s)
    if not m:
        return ""
    return f"{'un' if m.group(1) else ''}stable_{'core' if m.group(2).lower() == 'core' else 'ext'}"


def classify(rel: Path) -> Tuple[str, str]:
    """(project, group) guessed from the relative path."""
    dirs = list(rel.parts[:-1])
    stem = rel.name
    for suffix in (".gz", ".smt2"):
        stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    project = group = ""
    for part, is_dir in [(d, True) for d in dirs] + [(stem, False)]:
        if not group and GROUP_RE.search(part):
            group = _norm_group(part)
            continue
        if not project:
            m = PROJECT_RE.search(part)
            if m:
                # keep the WHOLE folder name (e.g. D_KOMODO vs S_KOMODO are different projects)
                project = part.lower() if is_dir else m.group(1).lower()
    return project or "default", group


def prepare_mariposa(source, out_dir, log=print) -> Path:
    src = Path(source).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.smt2"))
    if not files:
        gz = sorted(src.rglob("*.smt2.gz"))
        if gz:
            log(f"no plain .smt2 files; decompressing {len(gz)} .smt2.gz files into {out}/extracted ...")
            for p in gz:
                dest = out / "extracted" / p.relative_to(src).with_suffix("")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(p, "rb") as fi, open(dest, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            src = out / "extracted"
            files = sorted(src.rglob("*.smt2"))
    if not files:
        raise SystemExit(
            f"No .smt2 files found under {source}.\n"
            "Check that the folder is the 'benchmarks' folder of mariposa-data and that the download finished "
            "(run:  find {0} -type f | head -20 )".format(source)
        )
    rows: List[Dict] = []
    seen: Dict[str, str] = {}
    dups = 0
    ranked = sorted(files, key=lambda p: (GROUP_ORDER.get(classify(p.relative_to(src))[1], 9), str(p)))
    for p in ranked:
        h = hashlib.sha1(p.read_bytes()).hexdigest()
        if h in seen:
            dups += 1
            continue
        seen[h] = str(p)
        project, group = classify(p.relative_to(src))
        rows.append({"path": str(p), "project": project, "group": group})
    rows.sort(key=lambda r: r["path"])
    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "project", "group"])
        w.writeheader()
        w.writerows(rows)

    table: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        table[r["project"]][r["group"] or "?"] += 1
    groups = sorted({g for c in table.values() for g in c})
    log(f"\n{len(rows)} unique queries ({dups} exact duplicates dropped) -> {manifest}\n")
    log(f"{'project':22s} " + " ".join(f"{g:>14s}" for g in groups) + "   total")
    for proj in sorted(table):
        log(f"{proj:22s} " + " ".join(f"{table[proj][g]:14d}" for g in groups) + f"   {sum(table[proj].values()):5d}")
    if len(table) < 4:
        log("\nWARNING: fewer than 4 projects were recognised, so held-out-project evaluation will be weak. "
            "Copy this whole output to Claude.")
    if "?" in groups:
        log("\nNOTE: some queries have no recognised curated group (stable/unstable core/ext); the agreement check "
            "with the Mariposa classification will skip them.")
    return manifest

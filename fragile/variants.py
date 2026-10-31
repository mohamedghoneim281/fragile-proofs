"""Deterministic, meaning-preserving variants of an SMT query.

Variant ids look like ``"rename:3"``.  ``orig:0`` is the untouched query.
Everything is a pure function of (base_seed, query id, variant id), so a variant can
be regenerated later (for evidence bundles) without storing it.
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Sequence, Tuple

from .smt2 import Script, rename_symbols

FAMILIES = ("rename", "shuffle", "reseed", "mix")
ORIG = "orig:0"


def parse_variant(vid: str) -> Tuple[str, int]:
    fam, k = vid.split(":")
    return fam, int(k)


def derive_seed(base_seed: int, qid: str, vid: str, salt: str) -> int:
    h = hashlib.sha256(f"{base_seed}|{qid}|{vid}|{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 2) + 1


def feasible(script: Script, family: str) -> bool:
    if family == "rename":
        return script.can_rename
    if family == "shuffle":
        return script.can_shuffle
    if family == "reseed":
        return True
    if family == "mix":
        return script.can_rename or script.can_shuffle
    raise ValueError(f"unknown family {family!r}")


def planned_variants(script: Script, families: Sequence[str], k: int) -> List[str]:
    """``orig:0`` plus ``k`` variants of every family that can change this query."""
    out = [ORIG]
    for fam in families:
        if feasible(script, fam):
            out.extend(f"{fam}:{i}" for i in range(k))
    return out


def render_variant(script: Script, qid: str, vid: str, base_seed: int) -> Tuple[str, Optional[int]]:
    """Return (smt2 text, solver random seed or None)."""
    fam, _ = parse_variant(vid)
    if fam == "orig":
        return script.text, None
    text = script.text
    seed: Optional[int] = None
    if fam in ("shuffle", "mix"):
        text = script.shuffled_text(random.Random(derive_seed(base_seed, qid, vid, "shuffle")))
    if fam in ("rename", "mix"):
        mapping = script.rename_mapping(random.Random(derive_seed(base_seed, qid, vid, "rename")))
        text = rename_symbols(text, mapping)
    if fam in ("reseed", "mix"):
        seed = derive_seed(base_seed, qid, vid, "solver")
    return text, seed


def pool_or_ref(vid: str) -> str:
    """Split variants into a *pool* (what the strategies may test) and an independent
    *reference* set (what defines the ground-truth label): orig and even k -> pool, odd k -> ref."""
    fam, k = parse_variant(vid)
    if fam == "orig":
        return "pool"
    return "pool" if k % 2 == 0 else "ref"

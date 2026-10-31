"""Append-only measurement store (one JSON object per line) so long runs can be resumed."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Settings that must not change between sessions that write to the same store.
LOCKED_META = ("solver_id", "timeout_s", "base_seed", "success_mode", "rlimit", "mem_mb")


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.obs_path = self.root / "observations.jsonl"
        self._fh = None

    # ---- observations ------------------------------------------------------------
    def append(self, rec: Dict) -> None:
        if self._fh is None:
            self._fh = open(self.obs_path, "a", buffering=1)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def read(self) -> List[Dict]:
        out: List[Dict] = []
        if not self.obs_path.exists():
            return out
        with open(self.obs_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a half-written last line after a crash
        return out

    def latest(self) -> Dict[Tuple[str, str, int], Dict]:
        """Most recent record for every (query, variant, repeat); later lines win."""
        out: Dict[Tuple[str, str, int], Dict] = {}
        for r in self.read():
            out[(r["query"], r["variant"], r["repeat"])] = r
        return out

    def done_keys(self, retry_errors: bool = False) -> Set[Tuple[str, str, int]]:
        """Keys that need no (further) measurement. With ``retry_errors`` a key whose latest
        record is a solver 'error' counts as not done, so it is measured again (the old line stays in the log)."""
        return {k for k, r in self.latest().items() if not (retry_errors and r["outcome"] == "error")}

    # ---- json side files ----------------------------------------------------------
    def save_json(self, name: str, obj) -> None:
        tmp = self.root / (name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp, self.root / name)

    def load_json(self, name: str, default=None):
        p = self.root / name
        if not p.exists():
            return default
        with open(p) as fh:
            return json.load(fh)

    # ---- metadata -----------------------------------------------------------------
    def check_and_update_meta(self, new: Dict, force: bool = False) -> Dict:
        old = self.load_json("meta.json")
        if old and not force:
            bad = [k for k in LOCKED_META if k in old and old[k] != new.get(k)]
            if bad:
                raise SystemExit(
                    "This output folder already holds data measured with different settings: "
                    + ", ".join(f"{k} was {old[k]!r}, now {new.get(k)!r}" for k in bad)
                    + "\nUse a new --out folder (recommended) or --force."
                )
        merged = dict(old or {})
        merged.update(new)
        merged.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.save_json("meta.json", merged)
        return merged


def host_info() -> Dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu or platform.processor(),
        "cores": os.cpu_count(),
        "hostname": platform.node(),
    }

#!/usr/bin/env bash
# Copy the shareable outputs of a finished run into the repository's results/ folder,
# removing host names and absolute home-directory paths.
#
# Usage:  scripts/export_results.sh runs/main [results]
set -euo pipefail
SRC="${1:?usage: scripts/export_results.sh RUN_DIR [DEST_DIR]}"
DEST="${2:-results}"
mkdir -p "$DEST/raw" "$DEST/figures"

python3 - "$SRC" "$DEST" <<'PY'
import json, re, shutil, sys
from pathlib import Path

src, dest = Path(sys.argv[1]), Path(sys.argv[2])
home = re.compile(r"(/home/[^/\s\"]+|/Users/[^/\s\"]+)")

def clean(text: str) -> str:
    return home.sub("~", text)

meta = json.loads((src / "meta.json").read_text())
meta.get("host", {}).pop("hostname", None)
(dest / "raw" / "meta.json").write_text(clean(json.dumps(meta, indent=1)))
(dest / "meta.json").write_text(clean(json.dumps(meta, indent=1)))
(dest / "raw" / "queries.json").write_text(clean((src / "queries.json").read_text()))
shutil.copy(src / "observations.jsonl", dest / "raw" / "observations.jsonl")
for name in ("skipped.json", "confirmations.json"):
    if (src / name).exists():
        (dest / name).write_text(clean((src / name).read_text()))
res = src / "results"
if res.exists():
    for name in ("summary.md", "eval.json"):
        shutil.copy(res / name, dest / name)
    for png in res.glob("fig*.png"):
        shutil.copy(png, dest / "figures" / png.name)
print(f"exported to {dest}/ (raw log in {dest}/raw, replay with: python -m fragile evaluate {dest}/raw --ablations)")
PY

# status report regenerated from the log, with home-directory paths removed
python3 -m fragile status "$SRC" | sed -E 's#(/home/[^/ ]+|/Users/[^/ ]+)#~#g' > "$DEST/status.txt"

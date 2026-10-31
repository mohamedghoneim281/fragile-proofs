"""Regenerate the report figures from results/eval.json.

Usage (from the repository root):  python paper/make_figures.py
Writes PDF+PNG to paper/figs/ and PNG copies to results/figures/.
"""
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
ROOT = Path(__file__).resolve().parent.parent
e = json.loads((ROOT / "results" / "eval.json").read_text())
S = e["strategies"]
H = np.array(e["hours"])
out = ROOT / "paper" / "figs"
out.mkdir(parents=True, exist_ok=True)
col = {"uniform": "#444444", "random": "#9a9a9a", "runtime": "#d98e04", "adaptive": "#1f77b4", "adaptive-tuned": "#d62728"}
lab = {"uniform": "uniform", "random": "random", "runtime": "runtime-priority", "adaptive": "adaptive"}


def reach(y, t):
    i = np.where(np.array(y) >= t - 1e-12)[0]
    return float(H[i[0]])


# --- Figure: detection curves and cost to reach a recall level
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
main = list(lab)
for n in main:
    y, lo, hi = (np.array(S[n][k]) for k in ("curve_mean", "curve_lo", "curve_hi"))
    ax[0].plot(H, y, color=col[n], lw=1.8, label=lab[n])
    ax[0].fill_between(H, lo, hi, color=col[n], alpha=0.13, lw=0)
ax[0].set_xlim(0, 3.2)
ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("solver CPU time spent (hours)")
ax[0].set_ylabel("recall of detectable fragile queries")
ax[0].grid(alpha=0.25)
ax[0].legend(frameon=False, loc="lower right")
ax[0].set_title("(a) detection vs. CPU time", fontsize=9)
rs, w = [0.5, 0.8, 0.9], 0.2
x = np.arange(len(rs))
for k, n in enumerate(main):
    v = [reach(S[n]["curve_mean"], r) for r in rs]
    ax[1].bar(x + (k - 1.5) * w, v, w, color=col[n])
    for xi, vi in zip(x + (k - 1.5) * w, v):
        ax[1].text(xi, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=6.5, rotation=90)
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"{int(r * 100)}%" for r in rs])
ax[1].set_xlabel("target recall")
ax[1].set_ylabel("CPU hours needed")
ax[1].set_ylim(0, 2.8)
ax[1].set_title("(b) cost to reach a recall level", fontsize=9)
fig.tight_layout()
fig.savefig(out / "fig_main.pdf")
fig.savefig(out / "fig_main.png", dpi=200)
plt.close(fig)

# --- Figure: AUDC of every strategy and ablation
order = ["uniform", "random", "runtime", "adaptive", "adaptive-tuned", "adaptive-nopool", "adaptive-noboost", "adaptive-noexplore"]
nm = {"uniform": "uniform", "random": "random", "runtime": "runtime-priority", "adaptive": "adaptive",
      "adaptive-tuned": "adaptive, tuned on held-out folds", "adaptive-nopool": "adaptive, no project pooling",
      "adaptive-noboost": "adaptive, no slowness boost", "adaptive-noexplore": "adaptive, no exploration"}
fig, ax = plt.subplots(figsize=(6.2, 3.1))
for i, n in enumerate(order):
    m, lo, hi = S[n]["audc_mean"], S[n]["audc_p2.5"], S[n]["audc_p97.5"]
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=col.get(n, "#6fa8dc"), capsize=3, ms=5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([nm[n] for n in order])
ax.invert_yaxis()
ax.axvline(S["uniform"]["audc_mean"], color="#444444", lw=0.8, ls="--")
ax.set_xlabel("AUDC (area under normalised detection curve); bars: 2.5--97.5 percentile over %d seeds" % e["settings"]["seeds"])
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(out / "fig_audc.pdf")
fig.savefig(out / "fig_audc.png", dpi=200)
plt.close(fig)

# --- Figure: agreement with the curated groups
x = e["group_crosstab"]
groups = ["stable_core", "stable_ext", "unstable_ext", "unstable_core"]
gl = {"stable_core": "stable\ncore", "stable_ext": "stable\next", "unstable_ext": "unstable\next", "unstable_core": "unstable\ncore"}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
bot = np.zeros(len(groups))
for k, c, l in [("pure", "#c9d3e0", "reference-pure"), ("borderline", "#f2c879", "borderline"), ("fragile", "#c0392b", "fragile")]:
    v = np.array([x[g][k] / sum(x[g].values()) for g in groups])
    ax.bar(range(len(groups)), v, bottom=bot, color=c, label=l, width=0.65)
    for i, (vi, bi, g) in enumerate(zip(v, bot, groups)):
        if vi > 0.06:
            ax.text(i, bi + vi / 2, f"{x[g][k]}", ha="center", va="center", fontsize=7.5, color="black" if k != "fragile" else "white")
    bot += v
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{gl[g]}\n(n={sum(x[g].values())})" for g in groups], fontsize=8)
ax.set_ylabel("share of queries")
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.12))
fig.tight_layout()
fig.savefig(out / "fig_groups.pdf")
fig.savefig(out / "fig_groups.png", dpi=200)
plt.close(fig)

dest = ROOT / "results" / "figures"
dest.mkdir(parents=True, exist_ok=True)
for png in out.glob("*.png"):
    shutil.copy(png, dest / png.name)
print("figures written to", out)

"""Plot training progress from metrics_log.jsonl.

Usage:
    uv run python -m scripts.plot_training_progress \
        --metrics data/exp9_v3_extended/metrics_log.jsonl \
        --output data/exp9_v3_extended/training_progress.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Epsilon cycle boundaries (gen where each resume started, total gens for that resume)
CYCLES = [
    {"label": "Cycle 1", "start": 1,   "end": 20,  "color": "#d4e6f1"},
    {"label": "Cycle 2", "start": 21,  "end": 40,  "color": "#d5f5e3"},
    {"label": "Cycle 3", "start": 57,  "end": 100, "color": "#fdebd0"},
    {"label": "Cycle 4", "start": 101, "end": 150, "color": "#f9ebea"},
    {"label": "Cycle 5", "start": 151, "end": 200, "color": "#e8daef"},
    {"label": "Cycle 6", "start": 201, "end": 250, "color": "#d4e6f1"},
    {"label": "Cycle 7", "start": 251, "end": 300, "color": "#d5f5e3"},
]

# Epsilon schedule: given resume start gen and total gens, compute eps at each gen
EPS_START = 0.868
EPS_END   = 0.051

def eps_at_gen(gen, total_gens):
    progress = (gen - 1) / max(total_gens - 1, 1)
    return EPS_START + progress * (EPS_END - EPS_START)

# Map each gen to its cycle's total_gens
RESUME_SCHEDULE = [
    (1,   20,  20),
    (21,  40,  40),
    (41,  56,  60),   # killed early
    (57,  100, 100),
    (101, 150, 150),
    (151, 200, 200),
    (201, 250, 250),
    (251, 300, 300),
]

def get_epsilon(gen):
    for start, end, total in RESUME_SCHEDULE:
        if start <= gen <= end:
            return eps_at_gen(gen, total)
    return EPS_END


def load_metrics(path: Path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def smooth(vals, window=5):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window // 2)
        hi = min(len(vals), i + window // 2 + 1)
        out.append(np.mean(vals[lo:hi]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, default=Path("data/exp9_v3_extended/metrics_log.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/exp9_v3_extended/training_progress.png"))
    p.add_argument("--smooth", type=int, default=3, help="Smoothing window")
    args = p.parse_args()

    data = load_metrics(args.metrics)
    gens  = [d["generation"] for d in data]
    solo  = [d.get("eval/best_solo", float("nan")) for d in data]
    col   = [d.get("behavior/col_matches", float("nan")) for d in data]
    rev   = [d.get("behavior/rev_replace", float("nan")) for d in data]
    rcm   = [d.get("behavior/rev_col_match", float("nan")) for d in data]
    eps   = [get_epsilon(g) for g in gens]

    solo_s = smooth(solo, args.smooth)
    col_s  = smooth(col,  args.smooth)
    rev_s  = smooth(rev,  args.smooth)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Exp 10: Cyclic Epsilon Annealing — Training Progress", fontsize=13, fontweight="bold")

    # Shade cycle regions
    for ax in axes:
        for cyc in CYCLES:
            ax.axvspan(cyc["start"], min(cyc["end"], max(gens)), alpha=0.25,
                       color=cyc["color"], zorder=0)

    # --- Panel 1: Solo score + epsilon ---
    ax1 = axes[0]
    ax1b = ax1.twinx()
    ax1.plot(gens, solo, color="#aaaaaa", alpha=0.4, linewidth=0.8, zorder=1)
    ax1.plot(gens, solo_s, color="#2471a3", linewidth=2.0, label="Best solo [R,H,R]", zorder=2)
    ax1.axhline(14.0,  color="#e74c3c", linestyle="--", linewidth=1.2, label="Base heuristic (14.0)")
    ax1.axhline(10.52, color="#e67e22", linestyle="--", linewidth=1.2, label="Improved heuristic [R,H,R] (10.52)")
    ax1.axhline(8.10,  color="#27ae60", linestyle="--", linewidth=1.2, label="Improved heuristic [R,R,R] (8.10)")
    ax1b.plot(gens, eps, color="#8e44ad", linewidth=1.2, linestyle=":", alpha=0.7, label="Epsilon")
    ax1.set_ylabel("Score (lower is better)")
    ax1b.set_ylabel("Epsilon", color="#8e44ad")
    ax1b.tick_params(axis="y", labelcolor="#8e44ad")
    ax1b.set_ylim(0, 1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5,
               loc="upper left", bbox_to_anchor=(1.08, 1), borderaxespad=0)
    ax1.set_ylim(6, 20)
    ax1.invert_yaxis()

    # --- Panel 2: col_matches + rev_replace ---
    ax2 = axes[1]
    ax2.plot(gens, col, color="#aaaaaa", alpha=0.4, linewidth=0.8)
    ax2.plot(gens, col_s, color="#1a9850", linewidth=2.0, label="col_matches")
    ax2.plot(gens, rev_s, color="#d73027", linewidth=2.0, label="rev_replace")
    ax2.plot(gens, rcm,   color="#fc8d59", linewidth=1.2, alpha=0.7, label="rev_col_match")
    ax2.axhline(0.53, color="#1a9850", linestyle="--", linewidth=1.0, alpha=0.6, label="Heuristic col (0.53)")
    ax2.axhline(0.70, color="#1a9850", linestyle=":",  linewidth=1.0, alpha=0.6, label="Improved col (0.70)")
    ax2.axhline(0.33, color="#d73027", linestyle="--", linewidth=1.0, alpha=0.6, label="Improved rev (0.33)")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.08, 1), borderaxespad=0)

    # --- Panel 3: epsilon only (cleaner view of cycle structure) ---
    ax3 = axes[2]
    ax3.plot(gens, eps, color="#8e44ad", linewidth=2.0, label="Epsilon")
    ax3.fill_between(gens, eps, alpha=0.15, color="#8e44ad")
    ax3.set_ylabel("Epsilon")
    ax3.set_xlabel("Generation")
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.08, 1), borderaxespad=0)

    # Cycle labels on bottom panel
    for cyc in CYCLES:
        mid = (cyc["start"] + min(cyc["end"], max(gens))) / 2
        if mid <= max(gens):
            ax3.text(mid, 0.92, cyc["label"], ha="center", va="top",
                     fontsize=7, color="#555555")

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

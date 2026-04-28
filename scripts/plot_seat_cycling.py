"""Visualize seat-cycling results from scripts/seat_cycling.py output.

Usage:
    uv run python -m scripts.plot_seat_cycling \\
        --results data/seat_cycling_exp14_vs_lookahead.txt \\
        --output data/figures/seat_cycling_exp14_vs_lookahead.png \\
        --title "Exp14 DQN vs Lookahead (L,D,I,R)"
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_results(path: str):
    """Parse seat_cycling.py output.

    Returns:
        roster: list of labels in roster order
        summary: {label: (avg_score, win_rate)}
        per_seat: {label: {seat_idx: [(score, win_rate), ...]}}
    """
    summary: dict[str, tuple[float, float]] = {}
    per_seat: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    roster: list[str] = []

    in_summary = False
    in_breakdown = False

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        # Roster line: "=== Matchup: L,D,I,R (4 players) ==="
        m = re.match(r"=== Matchup: ([\w,]+) \(", line)
        if m:
            roster = m.group(1).split(",")

        if "Per-label summary" in line:
            in_summary = True
            in_breakdown = False
            continue
        if "Per-seating breakdown" in line:
            in_summary = False
            in_breakdown = True
            continue

        if in_summary:
            # "    L                9.113     48.3%"
            m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)%", line)
            if m:
                label = m.group(1)
                score = float(m.group(2))
                win_rate = float(m.group(3)) / 100.0
                summary[label] = (score, win_rate)

        if in_breakdown:
            # "    L,D,I,R   7.27/78.0%  9.97/19.6%  ..."
            if "/" not in line:
                continue
            m = re.match(r"\s+([\w,]+)\s+(.*)", line)
            if not m:
                continue
            seating = m.group(1).split(",")
            cells = re.findall(r"([\d.]+)/([\d.]+)%", m.group(2))
            for seat_idx, (score_s, win_s) in enumerate(cells):
                lbl = seating[seat_idx]
                per_seat[lbl][seat_idx].append(
                    (float(score_s), float(win_s) / 100.0)
                )

    return roster, summary, per_seat


def seat_matrix(per_seat, labels, n_seats=4):
    """Build (n_labels, n_seats) matrices for avg score and avg win rate."""
    scores = np.full((len(labels), n_seats), np.nan)
    wins = np.full((len(labels), n_seats), np.nan)
    for i, label in enumerate(labels):
        for seat in range(n_seats):
            vals = per_seat.get(label, {}).get(seat, [])
            if vals:
                scores[i, seat] = np.mean([v[0] for v in vals])
                wins[i, seat] = np.mean([v[1] for v in vals])
    return scores, wins


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "L": "#2196F3",   # blue
    "D": "#FF5722",   # deep orange
    "I": "#4CAF50",   # green
    "B": "#9C27B0",   # purple
    "B2": "#9C27B0",
    "B3": "#9C27B0",
    "R": "#9E9E9E",   # grey
    "D1": "#FF5722",
    "D2": "#FF9800",
}

LABEL_NAMES = {
    "L": "Lookahead",
    "D": "DQN (Exp14)",
    "D1": "DQN #1",
    "D2": "DQN #2",
    "I": "Improved heuristic",
    "B": "Bayes",
    "R": "Random",
}


def plot(roster, summary, per_seat, title: str, output: str):
    labels_ordered = sorted(summary, key=lambda l: summary[l][0])  # sort by score
    non_random = [l for l in labels_ordered if l != "R"]
    all_labels = labels_ordered

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                          left=0.07, right=0.97, top=0.91, bottom=0.09)

    ax_score = fig.add_subplot(gs[0, 0])
    ax_win   = fig.add_subplot(gs[0, 1])
    ax_sv    = fig.add_subplot(gs[0, 2])
    ax_ss    = fig.add_subplot(gs[1, 0])
    ax_sw    = fig.add_subplot(gs[1, 1])
    ax_h2h   = fig.add_subplot(gs[1, 2])

    colors = [COLORS.get(l, "#607D8B") for l in all_labels]
    names = [LABEL_NAMES.get(l, l) for l in all_labels]

    # --- Panel 1: avg score summary ---
    scores_summary = [summary[l][0] for l in all_labels]
    bars = ax_score.barh(names, scores_summary, color=colors, edgecolor="white", height=0.6)
    ax_score.set_xlabel("Avg score / hole (lower = better)")
    ax_score.set_title("Average score")
    ax_score.invert_xaxis()
    for bar, val in zip(bars, scores_summary):
        ax_score.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                      f"{val:.2f}", va="center", fontsize=8)

    # --- Panel 2: win rate summary ---
    wins_summary = [summary[l][1] * 100 for l in all_labels]
    bars = ax_win.barh(names, wins_summary, color=colors, edgecolor="white", height=0.6)
    ax_win.set_xlabel("Win rate (%)")
    ax_win.set_title("Win rate")
    chance = 100 / len(roster)
    ax_win.axvline(chance, color="black", linestyle="--", linewidth=0.8, label=f"Chance ({chance:.0f}%)")
    ax_win.legend(fontsize=7)
    for bar, val in zip(bars, wins_summary):
        ax_win.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)

    # --- Panel 3: score vs win rate scatter ---
    for l in all_labels:
        s, w = summary[l]
        c = COLORS.get(l, "#607D8B")
        ax_sv.scatter(w * 100, s, color=c, s=120, zorder=3)
        ax_sv.annotate(LABEL_NAMES.get(l, l), (w * 100, s),
                       textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax_sv.axvline(chance, color="black", linestyle="--", linewidth=0.8)
    ax_sv.set_xlabel("Win rate (%)")
    ax_sv.set_ylabel("Avg score / hole")
    ax_sv.set_title("Score vs win rate")
    ax_sv.invert_yaxis()

    # --- Panels 4 & 5: per-seat score and win rate ---
    seats = list(range(4))
    seat_labels = [f"Seat {i}" for i in seats]
    score_mat, win_mat = seat_matrix(per_seat, non_random)

    for i, label in enumerate(non_random):
        c = COLORS.get(label, "#607D8B")
        name = LABEL_NAMES.get(label, label)
        ax_ss.plot(seats, score_mat[i], marker="o", color=c, label=name, linewidth=1.8)
        ax_sw.plot(seats, win_mat[i] * 100, marker="o", color=c, label=name, linewidth=1.8)

    ax_ss.set_xticks(seats)
    ax_ss.set_xticklabels(seat_labels)
    ax_ss.set_ylabel("Avg score / hole")
    ax_ss.set_title("Score by seat position")
    ax_ss.invert_yaxis()
    ax_ss.legend(fontsize=7)

    ax_sw.set_xticks(seats)
    ax_sw.set_xticklabels(seat_labels)
    ax_sw.set_ylabel("Win rate (%)")
    ax_sw.set_title("Win rate by seat position")
    ax_sw.axhline(chance, color="black", linestyle="--", linewidth=0.8)
    ax_sw.legend(fontsize=7)

    # --- Panel 6: per-permutation gap (score) between top 2 labels by score ---
    top2 = labels_ordered[:2]
    l0, l1 = top2[0], top2[1]
    # Gather per-perm scores for each of the top 2 labels
    perm_scores_0: dict[tuple, float] = {}
    perm_scores_1: dict[tuple, float] = {}
    for seat_idx in range(4):
        for obs in per_seat.get(l0, {}).get(seat_idx, []):
            pass  # need seating key — rebuild from per_seat differently

    # Re-parse to get per-perm values for top2
    # Collect (perm_key, l0_score, l1_score) by reading per_seat seat-by-seat
    # Instead, use raw per_seat data: for each seat, each observation corresponds
    # to one permutation. Since each label appears in each seat 6 times in our
    # L,D,I,R 24-perm setup, zip observations together per seat index.
    perm_gaps = []
    for seat_idx in range(4):
        obs0 = per_seat.get(l0, {}).get(seat_idx, [])
        obs1 = per_seat.get(l1, {}).get(seat_idx, [])
        # These lists have different lengths in general; can't zip directly.
        # Just plot the distribution of (score_l1 - score_l0) across all observations.
        for s0, _ in obs0:
            for s1, _ in obs1:
                pass  # this is a cross-product, not right

    # Better: collect all score observations for l0 and l1, plot distributions
    all_scores_0 = [s for seat in range(4) for s, _ in per_seat.get(l0, {}).get(seat, [])]
    all_scores_1 = [s for seat in range(4) for s, _ in per_seat.get(l1, {}).get(seat, [])]

    n_bins = 12
    c0 = COLORS.get(l0, "#607D8B")
    c1 = COLORS.get(l1, "#607D8B")
    n0 = LABEL_NAMES.get(l0, l0)
    n1 = LABEL_NAMES.get(l1, l1)
    ax_h2h.hist(all_scores_0, bins=n_bins, alpha=0.6, color=c0, label=n0, density=True)
    ax_h2h.hist(all_scores_1, bins=n_bins, alpha=0.6, color=c1, label=n1, density=True)
    ax_h2h.axvline(np.mean(all_scores_0), color=c0, linestyle="--", linewidth=1.2)
    ax_h2h.axvline(np.mean(all_scores_1), color=c1, linestyle="--", linewidth=1.2)
    ax_h2h.set_xlabel("Avg score / hole (per seating)")
    ax_h2h.set_ylabel("Density")
    ax_h2h.set_title(f"Score distribution: {n0} vs {n1}")
    ax_h2h.legend(fontsize=7)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to seat_cycling output txt")
    p.add_argument("--output", required=True, help="Path for output PNG")
    p.add_argument("--title", default="Seat-cycling results", help="Figure title")
    args = p.parse_args()

    roster, summary, per_seat = parse_results(args.results)
    plot(roster, summary, per_seat, title=args.title, output=args.output)


if __name__ == "__main__":
    main()

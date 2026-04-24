"""Plot how the bayes belief evolves over a hole, comparing normal vs stacked decks.

For N games at each setting, snapshot the BayesBeliefTracker state at every
turn boundary (start of pid=0's turn). Aggregate per-turn averages of:

  - Per-rank multiset count (raw)
  - Per-rank marginal posterior P(unobserved card has rank R) = multiset/total
  - Total unobserved count

Then plot:

  Figure 1 (per_rank_marginal.png):
    Two subplots, one normal one stacked. X = turn index. Y = P(rank R).
    13 colored lines (one per rank). Low ranks (2/K/A) highlighted.

  Figure 2 (multiset_counts.png):
    Two subplots, one normal one stacked. X = turn. Y = avg multiset[R].
    Same structure as Figure 1 but raw counts.

  Figure 3 (total_unobserved.png):
    Single plot. X = turn. Y = avg total unobserved cards. Two lines
    (normal vs stacked).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.bayes_optimal import BayesBeliefTracker
from src.vectorized_golf import (
    NUM_CARDS,
    NUM_RANKS,
    RANK_SCORES,
    heuristic_stage0,
    improved_stage1,
    reset_games,
    step_stage0,
    step_stage1,
)


DEVICE = torch.device("cpu")

# Rank labels
RANK_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
# Three groups for plotting:
#   high-value (good for the player): A (1), K (0), 2 (-2). Lowest scores.
#   disaster: 10, J, Q. All score 10. Worst cards to be stuck with.
#   mid: 3..9. Middle range.
HIGH_VALUE_RANKS = [0, 11, 12]   # 2, K, A
DISASTER_RANKS = [8, 9, 10]      # 10, J, Q
MID_RANKS = [r for r in range(NUM_RANKS) if r not in HIGH_VALUE_RANKS and r not in DISASTER_RANKS]
GROUPS = {
    "high-value (2/K/A)": (HIGH_VALUE_RANKS, "tab:green"),
    "mid (3..9)": (MID_RANKS, "tab:gray"),
    "disaster (10/J/Q)": (DISASTER_RANKS, "tab:red"),
}


def collect_belief_evolution(
    stack_low_cards: bool,
    num_games: int,
    holes: int,
    n_players: int = 4,
):
    """Run N games (over `holes` holes) and at each pid=0 turn boundary
    snapshot the multiset.

    Returns:
        per_turn_multiset: dict[turn_idx -> list of (N,13) tensors]
        per_turn_total:    dict[turn_idx -> list of (N,)   tensors]
        per_turn_active:   dict[turn_idx -> list of (N,)   bool tensors]
    """
    torch.manual_seed(0)
    N = num_games
    tracker = BayesBeliefTracker(N, DEVICE)

    per_turn_multiset: Dict[int, List[torch.Tensor]] = defaultdict(list)
    per_turn_total: Dict[int, List[torch.Tensor]] = defaultdict(list)
    per_turn_active: Dict[int, List[torch.Tensor]] = defaultdict(list)

    for hole in range(holes):
        state = reset_games(
            N, DEVICE, n_players=n_players, stack_low_cards=stack_low_cards
        )
        tracker.reset()
        tracker.observe(state, my_player_id=0)

        bayes_turn = 0  # counts pid=0 turns within the hole
        for _ in range(60):
            if state.done.all():
                break
            for pid in range(n_players):
                if state.done.all():
                    break
                active = ~state.done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                active = ~state.done
                if not active.any():
                    break

                tracker.observe(state, my_player_id=0)

                if pid == 0:
                    # Snapshot at start of bayes player's turn (before action)
                    per_turn_multiset[bayes_turn].append(
                        tracker.multiset_by_rank().clone()
                    )
                    per_turn_total[bayes_turn].append(tracker.total().clone())
                    per_turn_active[bayes_turn].append(active.clone())

                state.current_stage.fill_(0)
                a0 = heuristic_stage0(state, pid)
                step_stage0(state, a0, pid)
                tracker.observe(state, my_player_id=0)
                if state.done.all():
                    break

                state.current_stage.fill_(1)
                a1 = improved_stage1(state, pid)
                step_stage1(state, a1, pid)
                tracker.observe(state, my_player_id=0)

                if pid == 0:
                    bayes_turn += 1

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly
                state.end_game_player = torch.where(
                    newly,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

    return per_turn_multiset, per_turn_total, per_turn_active


def aggregate(per_turn_multiset, per_turn_total, per_turn_active):
    """Compute per-turn averages of multiset[r] and P(rank=r) across active games.

    Returns:
        turns:           sorted list of turn indices
        avg_multiset:    np.ndarray shape (T, 13) -- avg multiset count per rank
        avg_p_per_rank:  np.ndarray shape (T, 13) -- avg multiset[r]/total per rank
        avg_total:       np.ndarray shape (T,)    -- avg unobserved count
        n_active:        np.ndarray shape (T,)    -- avg num active games per turn
    """
    turns = sorted(per_turn_multiset.keys())
    avg_multiset = np.zeros((len(turns), NUM_RANKS), dtype=np.float64)
    avg_p_per_rank = np.zeros((len(turns), NUM_RANKS), dtype=np.float64)
    avg_total = np.zeros(len(turns), dtype=np.float64)
    n_active = np.zeros(len(turns), dtype=np.float64)
    for i, t in enumerate(turns):
        ms_chunks = per_turn_multiset[t]
        tot_chunks = per_turn_total[t]
        act_chunks = per_turn_active[t]
        ms = torch.cat(ms_chunks, dim=0).float()        # (sum_N, 13)
        tot = torch.cat(tot_chunks, dim=0).float()      # (sum_N,)
        act = torch.cat(act_chunks, dim=0)              # (sum_N,) bool
        if act.any():
            ms_active = ms[act]
            tot_active = tot[act].clamp(min=1)
            avg_multiset[i] = ms_active.mean(dim=0).numpy()
            p_per_rank = ms_active / tot_active.unsqueeze(1)
            avg_p_per_rank[i] = p_per_rank.mean(dim=0).numpy()
            avg_total[i] = tot_active.mean().item()
            n_active[i] = int(act.sum().item())
    return np.array(turns), avg_multiset, avg_p_per_rank, avg_total, n_active


def plot_group_probability(turns_n, p_n, turns_s, p_s, n_active_n, n_active_s, out_path):
    """Plot P(next unobserved card is in group X) over turns, normal vs stacked.

    Groups: high-value (2/K/A), mid (3..9), disaster (10/J/Q).
    Each line is the SUM of per-rank marginals over the group, so the three
    lines per subplot sum to 1 at every turn.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, turns, p, n_active, title in [
        (axes[0], turns_n, p_n, n_active_n, "Normal deck"),
        (axes[1], turns_s, p_s, n_active_s, "Stacked deck (2/K/A at bottom)"),
    ]:
        valid = n_active > 50
        for label, (ranks, color) in GROUPS.items():
            group_p = p[:, ranks].sum(axis=1)
            ax.plot(turns[valid], group_p[valid], color=color, linewidth=2.5,
                    marker='o', label=label)
        # Reference: uniform-prior baseline (group_size / 13)
        for label, (ranks, color) in GROUPS.items():
            ax.axhline(len(ranks) / NUM_RANKS, color=color, linestyle=':',
                       alpha=0.5, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("pid=0 turn within hole")
        ax.set_ylabel("P(next unobserved card in group)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        "Group posterior: P(next unobserved card is in group X), by turn\n"
        "(dotted lines = uniform-prior baseline = group_size / 13)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_multiset_counts(turns_n, m_n, turns_s, m_s, n_active_n, n_active_s, out_path):
    """Plot fraction of group still unobserved over turns, normal vs stacked.

    Each group line is (sum of multiset counts in group) / (initial group size).
    All three groups start at 1.0 and decrease as their cards get observed.
    Normalized so the three groups are comparable on the same axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, turns, m, n_active, title in [
        (axes[0], turns_n, m_n, n_active_n, "Normal deck"),
        (axes[1], turns_s, m_s, n_active_s, "Stacked deck (2/K/A at bottom)"),
    ]:
        valid = n_active > 50
        for label, (ranks, color) in GROUPS.items():
            group_count = m[:, ranks].sum(axis=1)
            initial = len(ranks) * 4  # 4 cards per rank
            frac = group_count / initial
            ax.plot(turns[valid], frac[valid], color=color, linewidth=2.5,
                    marker='o', label=label)
        ax.set_title(title)
        ax.set_xlabel("pid=0 turn within hole")
        ax.set_ylabel("fraction of group still unobserved")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        "Belief: fraction of each group still unobserved, by turn",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_total_unobserved(turns_n, total_n, turns_s, total_s, n_active_n, n_active_s, out_path):
    """Plot total unobserved cards over turns."""
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_n = n_active_n > 50
    valid_s = n_active_s > 50
    ax.plot(turns_n[valid_n], total_n[valid_n], color='black', linewidth=2,
            marker='o', label='Normal deck')
    ax.plot(turns_s[valid_s], total_s[valid_s], color='darkorange', linewidth=2,
            marker='s', label='Stacked deck')
    ax.set_xlabel("pid=0 turn within hole")
    ax.set_ylabel("avg total unobserved cards")
    ax.set_title("Belief size: how fast does the player observe cards?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--holes", type=int, default=1, help="number of holes per run (each hole is one belief trajectory)")
    p.add_argument("--out-dir", type=str, default="data/figures/belief")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting normal deck...")
    n_ms, n_tot, n_act = collect_belief_evolution(
        stack_low_cards=False, num_games=args.games, holes=args.holes,
    )
    turns_n, m_n, p_n, total_n, na_n = aggregate(n_ms, n_tot, n_act)

    print("Collecting stacked deck...")
    s_ms, s_tot, s_act = collect_belief_evolution(
        stack_low_cards=True, num_games=args.games, holes=args.holes,
    )
    turns_s, m_s, p_s, total_s, na_s = aggregate(s_ms, s_tot, s_act)

    print("Plotting...")
    plot_group_probability(turns_n, p_n, turns_s, p_s, na_n, na_s,
                           out_dir / "group_probability.png")
    plot_multiset_counts(turns_n, m_n, turns_s, m_s, na_n, na_s,
                         out_dir / "multiset_counts.png")
    plot_total_unobserved(turns_n, total_n, turns_s, total_s, na_n, na_s,
                          out_dir / "total_unobserved.png")
    print("Done.")


if __name__ == "__main__":
    main()

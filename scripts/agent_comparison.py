"""Seat-cycled agent comparison with score distributions, rank distributions, and win rates.

Runs all agents together in every distinct seating permutation,
collecting per-hole scores, final card ranks, and per-hole winners. This eliminates
seat bias entirely.

Usage:
    # Single DQN (4-player: L, D, I, R)
    uv run python -m scripts.agent_comparison \
        --dqn-checkpoint data/exp11_cyclic/champion.pt \
        --games 1000 --holes 9 --seed 0

    # Two DQNs side-by-side (5-player: L, D1, D2, I, R)
    uv run python -m scripts.agent_comparison \
        --dqn1-checkpoint data/exp11_cyclic/champion.pt \
        --dqn1-name "DQN Exp11" \
        --dqn2-checkpoint data/exp14_win_bonus/gen_350/gen350_agent4.pt \
        --dqn2-name "DQN Exp14" \
        --games 1000 --holes 9 --seed 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.seat_cycling import SeatHandler
from src.vectorized_golf import (
    NUM_RANKS,
    compute_final_score,
    reset_games,
    step_stage0,
    step_stage1,
)

RANK_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

COLORS = {
    "L":  "#2471a3",
    "D":  "#e74c3c",
    "D1": "#e74c3c",
    "D2": "#e67e22",
    "I":  "#27ae60",
    "R":  "#95a5a6",
}


def unique_permutations(roster: List[str]) -> List[Tuple[str, ...]]:
    seen = set()
    out = []
    for perm in permutations(roster):
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


@torch.no_grad()
def collect_seat_cycled(
    roster: List[str],
    num_games: int,
    holes: int,
    device: torch.device,
    agent_names: Dict[str, str] | None = None,
) -> Dict[str, dict]:
    """Run all permutations, return {label: {scores, ranks, wins, holes_played}}."""
    perms = unique_permutations(roster)
    n_players = len(roster)
    N = num_games

    all_scores = defaultdict(list)
    all_ranks = defaultdict(list)
    win_counts = defaultdict(int)
    tie_counts = defaultdict(int)
    total_holes = defaultdict(int)

    names = agent_names or {}
    for pi, seating in enumerate(perms):
        print(f"  perm {pi+1}/{len(perms)}: {','.join(seating)}")
        handlers = [SeatHandler(label, seat, N, device) for seat, label in enumerate(seating)]

        for hole in range(holes):
            state = reset_games(N, device, n_players=n_players)
            for h in handlers:
                h.reset_for_hole(state)

            for _ in range(60):
                if state.done.all():
                    break

                for pid in range(n_players):
                    active = ~state.done
                    back_to_trigger = state.last_turn & (state.end_game_player == pid)
                    state.done = state.done | (back_to_trigger & active)
                    active = ~state.done
                    if not active.any():
                        break

                    handler = handlers[pid]

                    state.current_stage.fill_(0)
                    a0 = handler.stage0(state)
                    step_stage0(state, a0, pid)
                    for h in handlers:
                        h.observe(state)
                    if state.done.all():
                        break

                    state.current_stage.fill_(1)
                    a1 = handler.stage1(state)
                    step_stage1(state, a1, pid)
                    for h in handlers:
                        h.observe(state)

                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

            # Collect per-seat scores for this hole
            hole_scores = {}  # sid -> (N,) numpy
            for sid in range(n_players):
                label = seating[sid]
                scores = compute_final_score(state.player_cards[:, sid, :], device)
                ranks = (state.player_cards[:, sid, :] % NUM_RANKS).long()
                scores_np = scores.cpu().numpy()
                hole_scores[sid] = scores_np
                all_scores[label].append(scores_np)
                all_ranks[label].append(ranks.cpu().numpy().ravel())

            # Determine winner of each game in this hole (lowest score wins)
            score_matrix = np.stack([hole_scores[sid] for sid in range(n_players)], axis=1)  # (N, n_players)
            min_scores = score_matrix.min(axis=1, keepdims=True)  # (N, 1)
            is_winner = score_matrix == min_scores  # (N, n_players)
            n_winners = is_winner.sum(axis=1)  # (N,) -- >1 means tie

            for sid in range(n_players):
                label = seating[sid]
                sole_wins = (is_winner[:, sid] & (n_winners == 1)).sum()
                ties = (is_winner[:, sid] & (n_winners > 1)).sum()
                win_counts[label] += int(sole_wins)
                tie_counts[label] += int(ties)
                total_holes[label] += N

    labels = list(dict.fromkeys(roster))  # unique, order-preserving
    return {
        label: {
            "scores": np.concatenate(all_scores[label]),
            "ranks": np.concatenate(all_ranks[label]),
            "wins": win_counts[label],
            "ties": tie_counts[label],
            "total": total_holes[label],
        }
        for label in labels
    }


def plot_comparison(
    data: Dict[str, dict],
    roster: List[str],
    agent_names: Dict[str, str],
    output: Path,
    num_games: int,
    holes: int,
    n_perms: int,
) -> None:
    n_agents = len(roster)
    chance_pct = 100 / n_agents

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_score, ax_rank, ax_win = axes

    # --- Left: score distribution ---
    bins = np.arange(-10, 55, 1)
    for label in roster:
        d = data[label]
        name = agent_names[label]
        color = COLORS.get(label, "#607D8B")
        ax_score.hist(
            d["scores"], bins=bins, alpha=0.35, color=color,
            label=f"{name} (mean={d['scores'].mean():.2f})",
            density=True,
        )
        ax_score.axvline(
            d["scores"].mean(), color=color, linestyle="--",
            linewidth=1.5, alpha=0.8,
        )

    ax_score.set_xlabel("Score per hole")
    ax_score.set_ylabel("Density")
    ax_score.set_title("Per-hole score distribution")
    ax_score.legend(fontsize=8)
    ax_score.set_xlim(-10, 50)

    # --- Middle: card rank distribution ---
    x = np.arange(NUM_RANKS)
    bar_width = 0.8 / n_agents

    for i, label in enumerate(roster):
        d = data[label]
        name = agent_names[label]
        color = COLORS.get(label, "#607D8B")
        counts = np.bincount(d["ranks"], minlength=NUM_RANKS)[:NUM_RANKS]
        fractions = counts / counts.sum()
        ax_rank.bar(
            x + i * bar_width, fractions, bar_width,
            color=color, alpha=0.8, label=name,
        )

    ax_rank.set_xticks(x + bar_width * (n_agents - 1) / 2)
    ax_rank.set_xticklabels(RANK_LABELS)
    ax_rank.set_xlabel("Card rank")
    ax_rank.set_ylabel("Fraction of final layout cards")
    ax_rank.set_title("Rank distribution of kept cards")
    ax_rank.legend(fontsize=8)

    # --- Right: win rates ---
    labels_plot = [agent_names[l] for l in roster]
    win_rates = [data[l]["wins"] / data[l]["total"] * 100 for l in roster]
    tie_rates = [data[l]["ties"] / data[l]["total"] * 100 for l in roster]
    bar_colors = [COLORS.get(l, "#607D8B") for l in roster]

    bars_w = ax_win.bar(labels_plot, win_rates, color=bar_colors, alpha=0.8, label="Solo win")
    bars_t = ax_win.bar(labels_plot, tie_rates, bottom=win_rates, color=bar_colors, alpha=0.4, label="Tied win")

    for bar_w, bar_t, wr, tr in zip(bars_w, bars_t, win_rates, tie_rates):
        total = wr + tr
        ax_win.text(
            bar_w.get_x() + bar_w.get_width() / 2,
            total + 0.5,
            f"{total:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_win.set_ylabel("Win rate (%)")
    ax_win.set_title("Per-hole win rate (lowest score)")
    ax_win.axhline(chance_pct, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_win.text(len(roster) - 0.45, chance_pct + 0.4,
                f"chance={chance_pct:.0f}%", fontsize=7, color="gray")
    ax_win.legend(fontsize=8)
    ax_win.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"Seat-cycled comparison: {n_perms} permutations x {num_games} games x {holes} holes",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output}")


def _load_dqn(path: str, label: str, device: torch.device) -> None:
    from src.tournament import make_model, get_obs_fn
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    variant = cfg.get("model_variant", "v1")
    hidden_dim = cfg["hidden_dim"]
    embedding_dim = cfg.get("embedding_dim", 128)
    model = make_model(variant, embedding_dim, hidden_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    SeatHandler.dqn_registry[label] = (model, get_obs_fn(variant), device)
    print(f"Loaded {label} from {path} (variant={variant}, hidden={hidden_dim})")


def main():
    p = argparse.ArgumentParser(description="Seat-cycled agent comparison")
    # Single-DQN mode (legacy): --dqn-checkpoint → label D, roster L,D,I,R
    p.add_argument("--dqn-checkpoint", type=str, default=None,
                   help="Single DQN checkpoint (label D). Mutually exclusive with --dqn1/2-checkpoint.")
    # Two-DQN mode: --dqn1-checkpoint and --dqn2-checkpoint → labels D1/D2, roster L,D1,D2,I,R
    p.add_argument("--dqn1-checkpoint", type=str, default=None)
    p.add_argument("--dqn1-name", type=str, default="DQN 1")
    p.add_argument("--dqn2-checkpoint", type=str, default=None)
    p.add_argument("--dqn2-name", type=str, default="DQN 2")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="data/figures/agent_comparison.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    two_dqn_mode = args.dqn1_checkpoint is not None or args.dqn2_checkpoint is not None
    if two_dqn_mode:
        if not args.dqn1_checkpoint or not args.dqn2_checkpoint:
            raise SystemExit("Both --dqn1-checkpoint and --dqn2-checkpoint are required in two-DQN mode.")
        _load_dqn(args.dqn1_checkpoint, "D1", device)
        _load_dqn(args.dqn2_checkpoint, "D2", device)
        roster = ["L", "D1", "D2", "I", "R"]
        agent_names = {
            "L": "Lookahead",
            "D1": args.dqn1_name,
            "D2": args.dqn2_name,
            "I": "Improved Heuristic",
            "R": "Random",
        }
    else:
        if not args.dqn_checkpoint:
            raise SystemExit("Provide --dqn-checkpoint (single-DQN) or both --dqn1/2-checkpoint.")
        _load_dqn(args.dqn_checkpoint, "D", device)
        roster = ["L", "D", "I", "R"]
        agent_names = {"L": "Lookahead", "D": "DQN", "I": "Improved Heuristic", "R": "Random"}

    perms = unique_permutations(roster)
    print(f"Running {len(perms)} permutations x {args.games} games x {args.holes} holes...")

    data = collect_seat_cycled(roster, args.games, args.holes, device, agent_names)

    print(f"\nResults (seat-cycled, {len(perms)} perms x {args.games} games x {args.holes} holes):")
    print(f"  {'Agent':25s} {'Avg':>7} {'Std':>7} {'Win%':>7} {'Tie%':>7} {'Win+Tie%':>9}")
    print(f"  {'-'*63}")
    for label in roster:
        d = data[label]
        name = agent_names[label]
        wr = d["wins"] / d["total"] * 100
        tr = d["ties"] / d["total"] * 100
        print(f"  {name:25s} {d['scores'].mean():>7.3f} {d['scores'].std():>7.3f} {wr:>6.1f}% {tr:>6.1f}% {wr+tr:>8.1f}%")

    plot_comparison(data, roster, agent_names, Path(args.output), args.games, args.holes, len(perms))


if __name__ == "__main__":
    main()

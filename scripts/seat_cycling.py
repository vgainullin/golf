"""Seat-cycling head-to-head evaluation.

Compares Golf players (Bayes / improved heuristic / random) by enumerating
every distinct seat-permutation of a roster and aggregating scores by player
LABEL rather than by seat. This eliminates two well-known seat artifacts:

  1. First-seat advantage: seat 0 acts first and can trigger end-game first.
  2. Follower disadvantage: sitting after an efficient player is harder
     because they reveal less and consume the discard pile more usefully.

By averaging across all distinct seatings of the roster, every player label
occupies every seat with every neighbor configuration equally often.

Usage:

    uv run python -m scripts.seat_cycling --roster B,I,I,I --games-per-perm 1000 --holes 9
    uv run python -m scripts.seat_cycling --roster B,B,I,I --n-players 4
    uv run python -m scripts.seat_cycling --roster B,I,R,R --n-players 4

Roster labels:
    B = Bayes (belief-aware improved heuristic)
    I = Improved heuristic
    R = Random
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Tuple

import torch

from src.bayes_optimal import (
    BayesBeliefTracker,
    bayes_stage0,
    bayes_stage1,
    bayes_v2_stage0,
    bayes_v3_stage0,
)
from src.vectorized_golf import (
    compute_final_score,
    heuristic_stage0,
    improved_stage1,
    random_stage0,
    random_stage1,
    reset_games,
    step_stage0,
    step_stage1,
)


# ---------------------------------------------------------------------------
# Seat handler
# ---------------------------------------------------------------------------


class SeatHandler:
    """Wraps a player at a fixed seat with its own state (e.g. belief tracker
    for Bayes). Exposes uniform stage0/stage1 callables and a per-hole reset.

    Labels:
      B  -- naive belief-augmented player (bayes_stage0 + bayes_stage1).
            Known to be worse than IH; kept for ablation.
      B2 -- IH + hidden-card column-match check on stage 0 (bayes_v2_stage0
            with `improved_stage1`). Strict superset of IH.
      I  -- improved heuristic (heuristic_stage0 + improved_stage1).
      R  -- random.
    """

    LABELS = ("B", "B2", "B3", "I", "R")

    # Tunable per-handler config
    b2_cutoff: float = float(4)
    b3_draw_override_threshold: float = 0.50

    def __init__(self, label: str, seat_idx: int, N: int, device: torch.device):
        if label not in self.LABELS:
            raise ValueError(f"Unknown label {label!r}; valid: {self.LABELS}")
        self.label = label
        self.seat = seat_idx
        self.N = N
        self.device = device

        if label in ("B", "B2", "B3"):
            self.tracker = BayesBeliefTracker(N, device)
        else:
            self.tracker = None

    def reset_for_hole(self, state) -> None:
        if self.tracker is not None:
            self.tracker.reset()
            self.tracker.observe(state, my_player_id=self.seat)

    def observe(self, state) -> None:
        """Called between every step so the bayes tracker accumulates info."""
        if self.tracker is not None:
            self.tracker.observe(state, my_player_id=self.seat)

    def stage0(self, state) -> torch.Tensor:
        if self.label == "B":
            self.tracker.observe(state, my_player_id=self.seat)
            return bayes_stage0(state, self.seat, self.tracker)
        elif self.label == "B2":
            self.tracker.observe(state, my_player_id=self.seat)
            return bayes_v2_stage0(
                state, self.seat, self.tracker,
                cutoff=SeatHandler.b2_cutoff,
            )
        elif self.label == "B3":
            self.tracker.observe(state, my_player_id=self.seat)
            return bayes_v3_stage0(
                state, self.seat, self.tracker,
                draw_override_threshold=SeatHandler.b3_draw_override_threshold,
            )
        elif self.label == "I":
            return heuristic_stage0(state, self.seat)
        else:  # R
            return random_stage0(state, self.seat)

    def stage1(self, state) -> torch.Tensor:
        if self.label == "B":
            self.tracker.observe(state, my_player_id=self.seat)
            return bayes_stage1(state, self.seat, self.tracker)
        elif self.label in ("B2", "B3"):
            # B2/B3 keep IH's stage 1 -- only stage 0 is augmented.
            return improved_stage1(state, self.seat)
        elif self.label == "I":
            return improved_stage1(state, self.seat)
        else:  # R
            return random_stage1(state, self.seat)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def run_seating(
    seating: Tuple[str, ...],
    num_games: int,
    holes: int,
    device: torch.device,
    stack_low_cards: bool = False,
) -> List[float]:
    """Run num_games games with the given seating and return per-seat avg
    score / hole (length n_players).
    """
    n_players = len(seating)
    N = num_games
    handlers = [SeatHandler(label, seat, N, device) for seat, label in enumerate(seating)]

    totals = [torch.zeros(N, dtype=torch.float32, device=device) for _ in range(n_players)]

    for hole in range(1, holes + 1):
        state = reset_games(N, device, n_players=n_players, stack_low_cards=stack_low_cards)
        for h in handlers:
            h.reset_for_hole(state)

        # Sufficient turn cap. With reshuffle even long games terminate quickly
        # because players keep revealing.
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
                # Every bayes-handler observes after every step (including
                # opponent steps) to capture cards passing through the discard.
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

        for sid in range(n_players):
            totals[sid] += compute_final_score(state.player_cards[:, sid, :], device)

    return [t.mean().item() / holes for t in totals]


# ---------------------------------------------------------------------------
# Matchup orchestration
# ---------------------------------------------------------------------------


def unique_permutations(roster: List[str]) -> List[Tuple[str, ...]]:
    """All distinct seat permutations of the roster."""
    seen = set()
    out = []
    for perm in permutations(roster):
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


def run_matchup(
    roster: List[str],
    num_games_per_perm: int,
    holes: int,
    device: torch.device,
    stack_low_cards: bool = False,
) -> Tuple[Dict[str, float], List[Tuple[Tuple[str, ...], List[float]]]]:
    """Run all distinct seat-permutations of the roster.

    Returns (label_to_avg_score_per_hole, per_perm_table).
    The per-perm table is a list of (seating_tuple, [per-seat scores]).
    """
    perms = unique_permutations(roster)
    label_scores: Dict[str, List[float]] = defaultdict(list)
    per_perm: List[Tuple[Tuple[str, ...], List[float]]] = []

    for seating in perms:
        seat_avgs = run_seating(
            seating, num_games_per_perm, holes, device,
            stack_low_cards=stack_low_cards,
        )
        per_perm.append((seating, seat_avgs))
        for seat_idx, label in enumerate(seating):
            label_scores[label].append(seat_avgs[seat_idx])

    label_means = {label: sum(v) / len(v) for label, v in label_scores.items()}
    return label_means, per_perm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    roster: List[str],
    label_means: Dict[str, float],
    per_perm: List[Tuple[Tuple[str, ...], List[float]]],
    num_games_per_perm: int,
    holes: int,
) -> None:
    n_players = len(roster)
    total_games = num_games_per_perm * len(per_perm)
    print(f"=== Matchup: {','.join(roster)} ({n_players} players) ===")
    print(f"  {len(per_perm)} distinct seatings x {num_games_per_perm} games "
          f"x {holes} holes = {total_games * holes} hole-instances per label-instance")
    print()
    print(f"  Per-label avg score / hole (lower = better):")
    # Sort labels by average score
    for label in sorted(label_means.keys(), key=lambda l: label_means[l]):
        n_inst = sum(1 for s in roster if s == label)
        n_obs = n_inst * len(per_perm)
        print(f"    {label}: {label_means[label]:6.3f}  "
              f"(averaged over {n_obs} per-perm role-instances)")
    print()

    print(f"  Per-seating breakdown:")
    print(f"    {'seating':<20s}  " + "  ".join(f"seat{i}" for i in range(n_players)))
    for seating, seat_avgs in per_perm:
        seat_str = " ".join(f"{s:5.2f}" for s in seat_avgs)
        print(f"    {','.join(seating):<20s}  {seat_str}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--roster",
        type=str,
        required=True,
        help="Comma-separated player labels (B/I/R). Length = n_players.",
    )
    p.add_argument("--games-per-perm", type=int, default=1000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--b2-cutoff",
        type=float,
        default=float(4),
        help="cutoff for B2 belief-aware take rule (default 4 = IH cutoff).",
    )
    p.add_argument(
        "--b3-draw-threshold",
        type=float,
        default=0.50,
        help="threshold for B3 draw-override (default 0.50). With 1.01 the override never fires.",
    )
    p.add_argument(
        "--stack-low-cards",
        action="store_true",
        help="Stack the deck so all rank 2/K/A cards are at the bottom of the deck.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    SeatHandler.b2_cutoff = args.b2_cutoff
    SeatHandler.b3_draw_override_threshold = args.b3_draw_threshold

    roster = [r.strip() for r in args.roster.split(",")]
    for label in roster:
        if label not in SeatHandler.LABELS:
            raise SystemExit(f"Unknown label {label!r}; valid: {SeatHandler.LABELS}")

    label_means, per_perm = run_matchup(
        roster, args.games_per_perm, args.holes, device,
        stack_low_cards=args.stack_low_cards,
    )
    if args.stack_low_cards:
        print("(stacked deck: 2s, Ks, As at bottom)")
    print_report(roster, label_means, per_perm, args.games_per_perm, args.holes)


if __name__ == "__main__":
    main()

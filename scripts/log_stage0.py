"""Log every stage-0 decision during a run, with the Bayes posterior
P(deck draw score < face card score) computed from a per-seat belief tracker.

Each row of the log corresponds to one (game, turn, player) decision and
captures:
    label              -- player kind (B / I / R)
    face_score         -- score of the discard top
    p_draw_lt_face     -- P(deck_draw_score < face_score | belief)
    rank_match         -- IH's revealed-rank-match flag (True iff face rank
                          matches a face-up card already in own layout)
    decision           -- 0 = took face card, 1 = drew from deck
    drawn_score        -- score of the card actually drawn (NaN if took face)
    multiset_total     -- size of the unobserved set at decision time

After running, the analysis script (or a notebook) can identify cases where
IH's hard threshold made an objectively wrong call relative to the posterior.

Usage:
    uv run python -m scripts.log_stage0 --roster I,I,I,I --games 500 --holes 9
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from src.bayes_optimal import BayesBeliefTracker, bayes_stage0, bayes_stage1
from src.vectorized_golf import (
    NUM_RANKS,
    RANK_SCORES,
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
# Posterior helpers
# ---------------------------------------------------------------------------


def p_draw_lt_face(tracker: BayesBeliefTracker, face_score: torch.Tensor) -> torch.Tensor:
    """For each game, return P(rank_score(deck_draw) < face_score | belief).

    Uses the seat's multiset over unobserved cards. Strict less-than (a deck
    draw with the same score as the face card is treated as a tie, not a win).
    """
    multiset = tracker.multiset_by_rank().float()  # (N, 13)
    total = tracker.total().float().clamp(min=1)   # (N,)
    rank_scores = RANK_SCORES.to(tracker.device)   # (13,)
    # mask[n, r] = 1 iff rank_scores[r] < face_score[n]
    mask = rank_scores.unsqueeze(0) < face_score.unsqueeze(1)  # (N, 13)
    n_lt = (multiset * mask.float()).sum(dim=1)  # (N,)
    return n_lt / total


def rank_match_against_revealed(state, player_id: int) -> torch.Tensor:
    """Returns (N,) bool: True iff face rank matches a revealed own card."""
    face_rank = (state.discard_top % NUM_RANKS).long()  # (N,)
    cards = state.player_cards[:, player_id, :]
    revealed = state.player_revealed[:, player_id, :]
    ranks = cards % NUM_RANKS
    revealed_ranks = torch.where(revealed, ranks, torch.full_like(ranks, -1))
    return (revealed_ranks == face_rank.unsqueeze(1)).any(dim=1)


# ---------------------------------------------------------------------------
# Roster -> player function
# ---------------------------------------------------------------------------


def make_player_fns(label: str):
    if label == "I":
        return heuristic_stage0, improved_stage1
    elif label == "R":
        return random_stage0, random_stage1
    elif label == "B":
        # Bayes still works but needs the tracker passed in -- caller handles.
        return None, None
    else:
        raise ValueError(f"Unknown label {label!r}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class LoggedRun:
    label: List[str]                 # length M -- player label per decision
    face_score: torch.Tensor         # (M,) float
    p_draw_lt: torch.Tensor          # (M,) float
    rank_match: torch.Tensor         # (M,) bool
    decision: torch.Tensor           # (M,) int8 -- 0 take, 1 draw
    drawn_score: torch.Tensor        # (M,) float -- NaN if took face
    multiset_total: torch.Tensor     # (M,) int

    def save(self, path: Path) -> None:
        torch.save(
            {
                "label": self.label,
                "face_score": self.face_score,
                "p_draw_lt": self.p_draw_lt,
                "rank_match": self.rank_match,
                "decision": self.decision,
                "drawn_score": self.drawn_score,
                "multiset_total": self.multiset_total,
            },
            path,
        )


def run_logged(
    roster: List[str],
    num_games: int,
    holes: int,
    device: torch.device,
) -> LoggedRun:
    n_players = len(roster)
    N = num_games

    # One belief tracker per seat (each maintains its own POV).
    trackers = [BayesBeliefTracker(N, device) for _ in range(n_players)]

    # Player functions per seat (None for B -- handled inline).
    fns = [make_player_fns(lbl) for lbl in roster]

    label_buf: List[str] = []
    face_buf: List[torch.Tensor] = []
    plt_buf: List[torch.Tensor] = []
    rm_buf: List[torch.Tensor] = []
    dec_buf: List[torch.Tensor] = []
    drawn_buf: List[torch.Tensor] = []
    total_buf: List[torch.Tensor] = []

    rank_scores = RANK_SCORES.to(device)

    for hole in range(holes):
        state = reset_games(N, device, n_players=n_players)
        for pid in range(n_players):
            trackers[pid].reset()
            trackers[pid].observe(state, my_player_id=pid)

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

                # ---- Belief observe at start of turn ----
                trackers[pid].observe(state, my_player_id=pid)

                # ---- Compute posterior P(draw < face) BEFORE action ----
                face_rank = (state.discard_top % NUM_RANKS).long()
                face_score = rank_scores[face_rank]  # (N,)
                p_lt = p_draw_lt_face(trackers[pid], face_score)  # (N,)
                rm = rank_match_against_revealed(state, pid)  # (N,)
                multiset_total = trackers[pid].total().clone()  # (N,)

                # ---- Player decision ----
                state.current_stage.fill_(0)
                if roster[pid] == "B":
                    a0 = bayes_stage0(state, pid, trackers[pid])
                else:
                    a0 = fns[pid][0](state, pid)

                # Capture the would-be drawn card BEFORE step_stage0 advances deck_ptr
                deck_card = state.deck[
                    torch.arange(N, device=device),
                    state.deck_ptr.long().clamp(max=51),
                ]
                drawn_score = rank_scores[(deck_card % NUM_RANKS).long()]  # (N,)
                # NaN where decision != draw
                drawn_score_logged = torch.where(
                    a0 == 1, drawn_score, torch.full_like(drawn_score, float("nan"))
                )

                # Subset to active games (rest are filler / done)
                act = active.cpu()
                if act.any():
                    label_buf.extend([roster[pid]] * int(act.sum().item()))
                    face_buf.append(face_score.cpu()[act])
                    plt_buf.append(p_lt.cpu()[act])
                    rm_buf.append(rm.cpu()[act])
                    dec_buf.append(a0.to(torch.int8).cpu()[act])
                    drawn_buf.append(drawn_score_logged.cpu()[act])
                    total_buf.append(multiset_total.cpu()[act])

                step_stage0(state, a0, pid)
                trackers[pid].observe(state, my_player_id=pid)
                if state.done.all():
                    break

                state.current_stage.fill_(1)
                if roster[pid] == "B":
                    a1 = bayes_stage1(state, pid, trackers[pid])
                else:
                    a1 = fns[pid][1](state, pid)
                step_stage1(state, a1, pid)
                trackers[pid].observe(state, my_player_id=pid)

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

    return LoggedRun(
        label=label_buf,
        face_score=torch.cat(face_buf),
        p_draw_lt=torch.cat(plt_buf),
        rank_match=torch.cat(rm_buf),
        decision=torch.cat(dec_buf),
        drawn_score=torch.cat(drawn_buf),
        multiset_total=torch.cat(total_buf),
    )


# ---------------------------------------------------------------------------
# Quick analysis printout
# ---------------------------------------------------------------------------


def print_summary(run: LoggedRun, roster: List[str]) -> None:
    M = len(run.label)
    print(f"=== Stage-0 decision log ===")
    print(f"  roster: {','.join(roster)}")
    print(f"  total decisions: {M}")
    print()

    # Per label
    label_counts = Counter(run.label)
    for lbl in sorted(label_counts):
        idx = torch.tensor([i for i, l in enumerate(run.label) if l == lbl])
        if len(idx) == 0:
            continue
        face = run.face_score[idx]
        plt = run.p_draw_lt[idx]
        rm = run.rank_match[idx]
        dec = run.decision[idx]
        drawn = run.drawn_score[idx]

        n = len(idx)
        n_take = int((dec == 0).sum().item())
        n_draw = int((dec == 1).sum().item())

        print(f"  --- {lbl} ({n} decisions) ---")
        print(f"    take rate:   {100*n_take/n:5.1f}%")
        print(f"    draw rate:   {100*n_draw/n:5.1f}%")
        print(f"    avg P(draw < face):    {plt.mean().item():.3f}")
        print(f"    avg face_score:        {face.mean().item():.2f}")

        # When player drew, did the drawn card actually beat the face?
        drew_mask = dec == 1
        if drew_mask.any():
            drew_face = face[drew_mask]
            drew_outcome = drawn[drew_mask]
            beat = (drew_outcome < drew_face).float().mean().item()
            print(f"    when drew: drawn_score < face_score in {100*beat:5.1f}% of cases")

        # Calibration: bin by p_draw_lt and check actual rate of "drawn < face" among drawers
        if drew_mask.any():
            print(f"    calibration (drawers only):")
            print(f"      {'p_lt bin':<14s}  {'n':>6s}  {'predicted':>10s}  {'observed':>10s}")
            for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                           (0.4, 0.5), (0.5, 1.01)]:
                bin_mask = drew_mask & (run.p_draw_lt[idx] >= lo) & (run.p_draw_lt[idx] < hi)
                if bin_mask.sum() == 0:
                    continue
                predicted = run.p_draw_lt[idx][bin_mask].mean().item()
                observed = (drawn[bin_mask] < face[bin_mask]).float().mean().item()
                print(f"      [{lo:.2f}, {hi:.2f}):  "
                      f"{int(bin_mask.sum().item()):>6d}  {predicted:>10.3f}  {observed:>10.3f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roster", type=str, default="I,I,I,I")
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to save the raw log (torch.save)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    roster = [r.strip() for r in args.roster.split(",")]

    run = run_logged(roster, args.games, args.holes, device)
    print_summary(run, roster)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        run.save(out)
        print(f"  saved raw log: {out}")


if __name__ == "__main__":
    main()

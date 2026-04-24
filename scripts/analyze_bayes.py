"""Instrumented eval of the belief-augmented Bayes player.

Measures what the belief actually buys us at decision time:

  1. **Deterministic column-match opportunities**: at every stage-1 decision,
     we hold a card with rank R; for each column with one revealed slot of
     rank R and one face-down slot, placing R on the face-down slot
     guarantees a 0-score column. This is "free" -- any heuristic with rank
     awareness sees it; no inference needed.

  2. **Inferred column-match opportunities** (the interesting ones): a column
     with BOTH slots face-down. If we place our held rank-R card on slot A,
     the column matches iff slot B turns out to have rank R. The posterior
     probability is `P(slot B has rank R | belief) = multiset[R] / total`.
     We track the max of this across columns per decision -- this is the
     "highest-confidence inference-based match opportunity available right
     now". A high value means the belief is identifying a real edge.

  3. **Belief evolution by turn**: average multiset size and average
     E[score(unknown)] as a function of turn index. Tells us how fast the
     posterior collapses.

  4. **Realized columns matches** at game end (final layouts).

Run:

    uv run python -m scripts.analyze_bayes --games 2000 --holes 9
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from src.bayes_optimal import (
    BayesBeliefTracker,
    SEAT_PRESETS,
    bayes_stage0,
    bayes_stage1,
    parse_eval_config,
)
from src.vectorized_golf import (
    NUM_RANKS,
    compute_final_score,
    count_column_matches,
    reset_games,
    step_stage0,
    step_stage1,
)


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------


@dataclass
class BayesStats:
    """All counters we accumulate during instrumented play."""
    # Per-decision counts (one entry per (game, stage1-decision))
    decisions: int = 0
    decisions_with_det_match: int = 0     # any deterministic column-match available
    det_match_total_columns: int = 0      # sum of #det-matchable columns over all decisions
    inferred_p_max_sum: float = 0.0       # sum of max P(match | both-hidden col) per decision
    inferred_p_thresholds: Dict[float, int] = field(
        default_factory=lambda: {0.10: 0, 0.20: 0, 0.30: 0, 0.50: 0}
    )
    decisions_with_any_both_hidden_col: int = 0

    # Belief evolution by turn index (turn = 0, 1, 2, ... per game)
    multiset_size_by_turn: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    e_unknown_by_turn: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # End-of-game
    final_col_matches: List[float] = field(default_factory=list)
    final_scores: List[float] = field(default_factory=list)


def record_stage1_decision(
    state, tracker: BayesBeliefTracker, player_id: int, stats: BayesStats
) -> None:
    """Compute belief-derived metrics at a stage-1 decision (held card present)."""
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    held = state.player_holding[:, player_id].long()  # (N,)
    held_valid = held >= 0
    if not held_valid.any():
        return
    # All valid via the loop above (we only call this on stage 1)
    held_rank = (held % NUM_RANKS).clamp(min=0)  # (N,)

    cards = state.player_cards[:, player_id, :].long()  # (N, 6)
    revealed = state.player_revealed[:, player_id, :]  # (N, 6)
    ranks = cards % NUM_RANKS  # (N, 6)

    multiset = tracker.multiset_by_rank()  # (N, 13)
    total = tracker.total().float().clamp(min=1)  # (N,)

    # P(face-down slot has held_rank) = multiset[held_rank] / total
    # (N,)
    n_at_held = multiset.gather(1, held_rank.unsqueeze(1)).squeeze(1).float()
    p_match_held = n_at_held / total  # (N,)

    det_match_count = torch.zeros(N, dtype=torch.long, device=device)
    inferred_p_max = torch.zeros(N, dtype=torch.float32, device=device)
    has_both_hidden = torch.zeros(N, dtype=torch.bool, device=device)

    # Active = the games we're actually counting (~done & valid held)
    active = (~state.done) & held_valid

    for col in range(3):
        a, b = col, col + 3
        rev_a = revealed[:, a]
        rev_b = revealed[:, b]
        rank_a = ranks[:, a]
        rank_b = ranks[:, b]

        both_rev = rev_a & rev_b
        one_rev_a = rev_a & (~rev_b)
        one_rev_b = rev_b & (~rev_a)
        both_hidden = (~rev_a) & (~rev_b)

        # Deterministic case: column has one revealed slot whose rank == held_rank,
        # the other slot is face-down. Placing held on the face-down slot gives a
        # column-match (regardless of what was face-down).
        det_a = one_rev_a & (rank_a == held_rank)
        det_b = one_rev_b & (rank_b == held_rank)
        det_match_count = det_match_count + det_a.long() + det_b.long()

        # Inferred case: both slots face-down. Place held on either slot ->
        # column matches iff the OTHER slot has held_rank. Posterior = p_match_held.
        col_inferred_p = torch.where(
            both_hidden, p_match_held, torch.zeros_like(p_match_held)
        )
        inferred_p_max = torch.maximum(inferred_p_max, col_inferred_p)
        has_both_hidden = has_both_hidden | both_hidden

    # Accumulate scalars over the active subset
    n_active = int(active.sum().item())
    if n_active == 0:
        return

    stats.decisions += n_active
    stats.decisions_with_det_match += int(((det_match_count > 0) & active).sum().item())
    stats.det_match_total_columns += int(det_match_count[active].sum().item())
    stats.inferred_p_max_sum += float(inferred_p_max[active].sum().item())
    stats.decisions_with_any_both_hidden_col += int(
        (has_both_hidden & active).sum().item()
    )
    for thr in stats.inferred_p_thresholds:
        stats.inferred_p_thresholds[thr] += int(
            ((inferred_p_max >= thr) & active).sum().item()
        )


def record_belief_snapshot(
    tracker: BayesBeliefTracker, turn_idx: int, active: torch.Tensor, stats: BayesStats
) -> None:
    """Snapshot belief size and E[unknown] at the start of a (player_id=0) turn."""
    if not active.any():
        return
    multiset_total = tracker.total().float()
    e_unknown = tracker.expected_unknown_score()
    stats.multiset_size_by_turn[turn_idx].append(
        float(multiset_total[active].mean().item())
    )
    stats.e_unknown_by_turn[turn_idx].append(
        float(e_unknown[active].mean().item())
    )


# ---------------------------------------------------------------------------
# Instrumented eval loop
# ---------------------------------------------------------------------------


def run_instrumented(
    opponent_specs: List[str],
    num_games: int,
    holes: int,
    device: torch.device,
) -> BayesStats:
    N = num_games
    tracker = BayesBeliefTracker(N, device)
    stats = BayesStats()

    seat_fns = [(None, None)]  # placeholder for seat 0 (handled inline)
    for spec in opponent_specs:
        seat_fns.append(SEAT_PRESETS[spec])

    for hole in range(1, holes + 1):
        state = reset_games(N, device)
        tracker.reset()
        tracker.observe(state, my_player_id=0)

        # Per-hole turn counter for belief snapshots (only player-0 turns)
        bayes_turn = 0

        for _ in range(30):
            if state.done.all():
                break

            for pid in range(4):
                active = ~state.done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                active = ~state.done
                if not active.any():
                    break

                if pid == 0:
                    # ---- bayes seat: instrumented ----
                    tracker.observe(state, my_player_id=0)
                    record_belief_snapshot(tracker, bayes_turn, active, stats)

                    state.current_stage.fill_(0)
                    a0 = bayes_stage0(state, pid, tracker)
                    step_stage0(state, a0, pid)
                    tracker.observe(state, my_player_id=0)
                    if state.done.all():
                        break

                    state.current_stage.fill_(1)
                    record_stage1_decision(state, tracker, pid, stats)
                    a1 = bayes_stage1(state, pid, tracker)
                    step_stage1(state, a1, pid)
                    tracker.observe(state, my_player_id=0)

                    bayes_turn += 1
                else:
                    s0_fn, s1_fn = seat_fns[pid]
                    state.current_stage.fill_(0)
                    a0 = s0_fn(state, pid)
                    step_stage0(state, a0, pid)
                    tracker.observe(state, my_player_id=0)
                    if state.done.all():
                        break
                    state.current_stage.fill_(1)
                    a1 = s1_fn(state, pid)
                    step_stage1(state, a1, pid)
                    tracker.observe(state, my_player_id=0)

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        # End-of-hole stats
        col_matches = count_column_matches(state, player_id=0)  # (N,)
        final_score = compute_final_score(state.player_cards[:, 0, :], device)  # (N,)
        stats.final_col_matches.extend(col_matches.float().tolist())
        stats.final_scores.extend(final_score.tolist())

    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(stats: BayesStats, num_games: int, holes: int, opponents: List[str]) -> None:
    print(f"=== Bayes player instrumented eval ===")
    print(f"  config: [B,{','.join(opponents)}]  games={num_games}  holes={holes}")
    print()

    n = stats.decisions
    print(f"Stage-1 decisions analyzed: {n}")
    print()

    print("Deterministic column-match opportunities (rank match between held card and a revealed slot in a half-revealed column)")
    pct = 100 * stats.decisions_with_det_match / max(n, 1)
    avg = stats.det_match_total_columns / max(n, 1)
    print(f"  decisions with at least one:    {stats.decisions_with_det_match:>8d}  ({pct:5.2f}%)")
    print(f"  avg # of det-match columns:     {avg:>8.4f}")
    print()

    print("Inferred column-match posteriors (both slots face-down -> P(other has held rank))")
    n_with = stats.decisions_with_any_both_hidden_col
    print(f"  decisions with any both-hidden col: {n_with:>8d}  ({100*n_with/max(n,1):5.2f}%)")
    print(f"  avg max-posterior P(match):         {stats.inferred_p_max_sum/max(n,1):>8.4f}")
    print(f"  avg max-posterior (cond. on >0):    "
          f"{stats.inferred_p_max_sum/max(n_with,1):>8.4f}")
    print(f"  decisions with max P(match) >= threshold:")
    for thr, count in sorted(stats.inferred_p_thresholds.items()):
        print(f"    p >= {thr:.2f}:   {count:>8d}  ({100*count/max(n,1):5.2f}%)")
    print()

    print("Belief collapse over Bayes-player turns (per game)")
    print(f"  {'turn':>5} {'avg multiset':>14} {'avg E[unk]':>12}")
    turns = sorted(stats.multiset_size_by_turn.keys())
    for t in turns:
        sizes = stats.multiset_size_by_turn[t]
        eus = stats.e_unknown_by_turn[t]
        if not sizes:
            continue
        # These were already per-step batch means, average across holes:
        m = sum(sizes) / len(sizes)
        e = sum(eus) / len(eus)
        print(f"  {t:>5d} {m:>14.2f} {e:>12.3f}")
    print()

    print("End-of-game outcomes (seat 0)")
    matches = stats.final_col_matches
    scores = stats.final_scores
    avg_m = sum(matches) / len(matches)
    avg_s = sum(scores) / len(scores)
    print(f"  avg column matches per game: {avg_m:.3f}  (max possible 3)")
    print(f"  avg final score per game:    {avg_s:.3f}")
    # Match distribution
    counts = [0, 0, 0, 0]
    for m in matches:
        counts[int(round(m))] += 1
    print(f"  match-count distribution:    "
          f"0: {100*counts[0]/len(matches):5.1f}%  "
          f"1: {100*counts[1]/len(matches):5.1f}%  "
          f"2: {100*counts[2]/len(matches):5.1f}%  "
          f"3: {100*counts[3]/len(matches):5.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--eval-config", type=str, default="R,H,R")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    opponents = parse_eval_config(args.eval_config)

    stats = run_instrumented(opponents, args.games, args.holes, device)
    print_report(stats, args.games, args.holes, opponents)


if __name__ == "__main__":
    main()

"""Evaluate heuristic variants in all-heuristic games."""
import argparse
from functools import partial
import torch
from src.vectorized_golf import (
    reset_games, step_stage0, step_stage1,
    compute_final_score,
    heuristic_stage0, heuristic_stage1,
    simple_stage0, simple_stage1,
    improved_stage1,
    random_stage0, random_stage1,
)


def run_eval(s0_fn, s1_fn, num_games: int, holes: int, device: torch.device) -> list:
    """Run all-same-heuristic 4-player games, return avg score/hole per seat."""
    N = num_games
    totals = [torch.zeros(N, dtype=torch.float32, device=device) for _ in range(4)]

    for hole in range(1, holes + 1):
        state = reset_games(N, device)

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

                state.current_stage.fill_(0)
                actions_s0 = s0_fn(state, pid)
                step_stage0(state, actions_s0, pid)
                if state.done.all():
                    break

                state.current_stage.fill_(1)
                actions_s1 = s1_fn(state, pid)
                step_stage1(state, actions_s1, pid)

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        for sid in range(4):
            totals[sid] += compute_final_score(state.player_cards[:, sid, :], device)

    return [t.mean().item() / holes for t in totals]


def run_mixed_eval(seat_fns: list, num_games: int, holes: int, device: torch.device) -> float:
    """Run games with per-seat (s0_fn, s1_fn) tuples. Returns seat-0 avg score/hole.

    Matches the DQN solo eval config: seat 0 = agent under test, others = opponents.
    """
    N = num_games
    total = torch.zeros(N, dtype=torch.float32, device=device)

    for hole in range(1, holes + 1):
        state = reset_games(N, device)

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

                s0_fn, s1_fn = seat_fns[pid]

                state.current_stage.fill_(0)
                actions_s0 = s0_fn(state, pid)
                step_stage0(state, actions_s0, pid)
                if state.done.all():
                    break

                state.current_stage.fill_(1)
                actions_s1 = s1_fn(state, pid)
                step_stage1(state, actions_s1, pid)

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        total += compute_final_score(state.player_cards[:, 0, :], device)

    return total.mean().item() / holes


def print_result(name, scores):
    avg = sum(scores) / 4
    std = (sum((s - avg) ** 2 for s in scores) / 4) ** 0.5
    print(f"  {name:16s} {scores[0]:6.2f}  {scores[1]:6.2f}  {scores[2]:6.2f}  {scores[3]:6.2f}  {avg:6.2f}  {std:5.2f}")
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--holes", type=int, default=9)
    args = p.parse_args()

    device = torch.device("cpu")
    G, H = args.games, args.holes

    # --- Fixed variants ---
    print(f"=== Fixed variants ({G} games, {H} holes) ===")
    print(f"{'':18s} seat0   seat1   seat2   seat3   avg     std")

    fixed = [
        ("random",       random_stage0,    random_stage1),
        ("rand+base_s1", random_stage0,    heuristic_stage1),
        ("rand+impr_s1", random_stage0,    improved_stage1),
        ("simple",       simple_stage0,    simple_stage1),
        ("base",         heuristic_stage0, heuristic_stage1),
        ("improved",     heuristic_stage0, improved_stage1),
    ]
    results = {}
    for name, s0, s1 in fixed:
        scores = run_eval(s0, s1, G, H, device)
        results[name] = print_result(name, scores)

    print()
    names = list(results.keys())
    for i in range(1, len(names)):
        delta = results[names[i - 1]] - results[names[i]]
        print(f"  {names[i-1]:16s} -> {names[i]:16s}: {delta:+.2f} pts")

    # --- Solo eval: [player, R, H, R] matching DQN tournament eval config ---
    print(f"\n=== Solo eval [player, R, H, R] ({G} games, {H} holes) ===")
    print(f"  Matches DQN tournament solo eval config exactly (seat 0 score only)")
    print(f"{'':18s}  seat0")

    rand = (random_stage0, random_stage1)
    heur = (heuristic_stage0, heuristic_stage1)
    solo_variants = [
        ("random",   (random_stage0,    random_stage1)),
        ("simple",   (simple_stage0,    simple_stage1)),
        ("base",     (heuristic_stage0, heuristic_stage1)),
        ("improved", (heuristic_stage0, improved_stage1)),
    ]
    solo_results = {}
    for name, agent in solo_variants:
        seat_fns = [agent, rand, heur, rand]
        score = run_mixed_eval(seat_fns, G, H, device)
        print(f"  {name:16s} {score:6.2f}")
        solo_results[name] = score

    print()
    solo_names = list(solo_results.keys())
    for i in range(1, len(solo_names)):
        delta = solo_results[solo_names[i - 1]] - solo_results[solo_names[i]]
        print(f"  {solo_names[i-1]:16s} -> {solo_names[i]:16s}: {delta:+.2f} pts")

    # --- Cutoff sweeps ---
    cutoffs = list(range(0, 12))

    for label, s0_fn, s1_fn in [
        ("simple (no column awareness)", simple_stage0, simple_stage1),
        ("base heuristic", heuristic_stage0, heuristic_stage1),
    ]:
        print(f"\n=== Cutoff sweep: {label} ({G} games, {H} holes) ===")
        print(f"{'':18s} seat0   seat1   seat2   seat3   avg     std")

        best_cutoff, best_avg = None, 1e6
        for c in cutoffs:
            s0 = partial(s0_fn, cutoff=c)
            s1 = partial(s1_fn, cutoff=c) if s1_fn is heuristic_stage1 else s1_fn
            scores = run_eval(s0, s1, G, H, device)
            avg = print_result(f"cutoff={c}", scores)
            if avg < best_avg:
                best_avg = avg
                best_cutoff = c

        print(f"\n  Best cutoff: {best_cutoff} ({best_avg:.2f} avg/hole)")


if __name__ == "__main__":
    main()

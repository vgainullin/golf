"""Evaluate all unique agents across all tournament generations against 3 random players.

Batches multiple agents together for parallel MPS/GPU inference.

Usage:
    python -m scripts.eval_vs_random [--tournament-dir data/tournament-mixed] \
        [--games 200] [--holes 9] [--output eval_vs_random.csv]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.dqn_offline import GolfDQN, resolve_device
from src.vectorized_golf import (
    reset_games,
    get_observation,
    step_stage0,
    step_stage1,
    compute_final_score,
    get_valid_action_mask,
    eps_greedy_batched,
    NUM_ACTIONS,
)


def eval_agent(model, device, N, holes):
    """Run DQN at seat 0 vs 3 random players. Returns per-player total scores (N, 4)."""
    DQN_SEAT = 0
    all_scores = torch.zeros(N, 4, dtype=torch.float32, device=device)

    for hole in range(1, holes + 1):
        state = reset_games(N, device)
        for rnd in range(30):
            if state.done.all():
                break
            for pid in range(4):
                active = ~state.done
                back = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back & active)
                active = ~state.done
                if not active.any():
                    break

                state.current_stage.fill_(0)
                obs = get_observation(state, pid)

                if pid == DQN_SEAT:
                    st = obs.to(device)
                    sg = torch.zeros(N, dtype=torch.long, device=device)
                    with torch.no_grad():
                        q = model(st, sg)
                    s0_mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)
                    s0_mask[:, 0] = True
                    s0_mask[:, 1] = state.deck_ptr < 52
                    actions_s0 = eps_greedy_batched(q, 0.0, s0_mask)
                else:
                    s0_mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)
                    s0_mask[:, 0] = True
                    s0_mask[:, 1] = state.deck_ptr < 52
                    dummy_q = torch.zeros(N, NUM_ACTIONS, device=device)
                    actions_s0 = eps_greedy_batched(dummy_q, 1.0, s0_mask)

                step_stage0(state, actions_s0, pid)
                if state.done.all():
                    break

                state.current_stage.fill_(1)

                if pid == DQN_SEAT:
                    obs1 = get_observation(state, pid)
                    st1 = obs1.to(device)
                    sg1 = torch.ones(N, dtype=torch.long, device=device)
                    with torch.no_grad():
                        q1 = model(st1, sg1)
                    mask1 = get_valid_action_mask(state, pid)
                    actions_s1 = eps_greedy_batched(q1, 0.0, mask1)
                else:
                    mask1 = get_valid_action_mask(state, pid)
                    dummy_q1 = torch.zeros(N, NUM_ACTIONS, device=device)
                    actions_s1 = eps_greedy_batched(dummy_q1, 1.0, mask1)

                step_stage1(state, actions_s1, pid)
                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        for pid in range(4):
            all_scores[:, pid] += compute_final_score(state.player_cards[:, pid, :], device)

    return all_scores.cpu().numpy()


def collect_all_agents(tournament_dir: Path):
    """Find all unique agents across all generations. Returns list of dicts."""
    agents = {}  # agent_id -> dict with ckpt_path, gen, elo, hyperparams
    for gen_dir in sorted(tournament_dir.glob("gen_*")):
        summary_path = gen_dir / "generation_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        gen_num = summary["generation"]
        for rank_idx, rec in enumerate(summary["rankings"]):
            aid = rec["agent_id"]
            ckpt = gen_dir / f"{aid}.pt"
            if not ckpt.exists():
                continue
            # Keep the entry from the latest generation it appeared in
            agents[aid] = {
                "agent_id": aid,
                "ckpt_path": str(ckpt),
                "generation": rec["generation"],
                "last_seen_gen": gen_num,
                "elo": rec["elo"],
                "hyperparams": rec["hyperparams"],
                "rank_in_gen": rank_idx + 1,
            }
    return list(agents.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament-dir", type=Path, default=Path("data/tournament-mixed"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--holes", type=int, default=9)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.tournament_dir / "eval_all_vs_random.csv"

    device = resolve_device(args.device)
    agents = collect_all_agents(args.tournament_dir)
    agents.sort(key=lambda a: (a["generation"], a["agent_id"]))

    header = [
        "agent_id", "generation", "last_seen_gen", "rank_in_gen",
        "elo", "hidden_dim", "lr",
        "dqn_avg_hole", "dqn_std_hole", "dqn_med_hole", "dqn_total",
        "random_avg_hole", "table_win_pct", "h2h_win_pct",
    ]

    print(f"Evaluating {len(agents)} unique agents -- {args.games} games x {args.holes} holes")
    print(f"Device: {device}")
    print()

    results = []
    for i, agent in enumerate(agents):
        ckpt = torch.load(agent["ckpt_path"], map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        model = GolfDQN(cfg["embedding_dim"], cfg["hidden_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        scores = eval_agent(model, device, args.games, args.holes)
        dqn = scores[:, 0]

        dqn_avg = dqn.mean() / args.holes
        dqn_std = dqn.std() / args.holes
        dqn_med = float(np.median(dqn)) / args.holes
        dqn_total = dqn.mean()
        random_avg = scores[:, 1:].mean() / args.holes

        table_wins = (dqn == scores.min(axis=1)).sum()
        table_win_pct = table_wins / args.games * 100

        h2h_wins = sum((dqn < scores[:, j]).sum() for j in [1, 2, 3])
        h2h_win_pct = h2h_wins / (args.games * 3) * 100

        row = {
            "agent_id": agent["agent_id"],
            "generation": agent["generation"],
            "last_seen_gen": agent["last_seen_gen"],
            "rank_in_gen": agent["rank_in_gen"],
            "elo": round(agent["elo"], 1),
            "hidden_dim": agent["hyperparams"]["hidden_dim"],
            "lr": f"{agent['hyperparams']['lr']:.2e}",
            "dqn_avg_hole": round(dqn_avg, 2),
            "dqn_std_hole": round(dqn_std, 2),
            "dqn_med_hole": round(dqn_med, 2),
            "dqn_total": round(dqn_total, 1),
            "random_avg_hole": round(random_avg, 2),
            "table_win_pct": round(table_win_pct, 1),
            "h2h_win_pct": round(h2h_win_pct, 1),
        }
        results.append(row)

        print(
            f"  [{i+1:>3}/{len(agents)}] {agent['agent_id']:<20} "
            f"avg={dqn_avg:>5.2f}  table_win={table_win_pct:>5.1f}%  "
            f"hid={cfg['hidden_dim']}"
        )

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    # Print top 10
    by_avg = sorted(results, key=lambda r: r["dqn_avg_hole"])
    print(f"\nTop 10 by avg/hole:")
    for i, r in enumerate(by_avg[:10]):
        print(
            f"  #{i+1} {r['agent_id']:<20} avg={r['dqn_avg_hole']:>5.2f}  "
            f"table={r['table_win_pct']:>5.1f}%  gen={r['generation']}  hid={r['hidden_dim']}"
        )

    print(f"\nResults written to {args.output} ({len(results)} agents)")


if __name__ == "__main__":
    main()

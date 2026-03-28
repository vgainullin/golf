"""DEPRECATED: Use scripts/eval_hof.py or scripts/eval_vs_random.py instead.

This module uses the old non-vectorized simulation loop (src.simulation).
The newer scripts use the vectorized engine (src.vectorized_golf) and support
v3 models via the tournament infrastructure.
"""

from __future__ import annotations

import argparse
import json
import warnings
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .simulation import Golf, Player, play_game
from .offline_agent import OfflineDQAgent


def evaluate_self_play(
    checkpoints: List[Path],
    agent_names: List[str],
    num_games: int,
    holes_per_game: int,
    rotate_positions: bool,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run self-play evaluation with player rotation.

    Args:
        checkpoints: List of checkpoint paths (1-4 agents)
        agent_names: Names for each agent
        num_games: Number of games to play
        holes_per_game: Holes per game
        rotate_positions: Whether to rotate positions between games
        device: Device for inference
        output_dir: Directory to save results

    Returns:
        Dict with evaluation results
    """
    if len(checkpoints) != len(agent_names):
        raise ValueError("Number of checkpoints must match number of names")

    if len(checkpoints) < 1 or len(checkpoints) > 4:
        raise ValueError("Must provide 1-4 checkpoints")

    # Suppress checkpoint loading warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*weights_only.*")

    # Load agents
    print(f"\nLoading {len(checkpoints)} agents...")
    agents = []
    for i, (checkpoint, name) in enumerate(zip(checkpoints, agent_names)):
        print(f"  [{i+1}] {name}: {checkpoint}")
        agents.append(OfflineDQAgent(checkpoint, device=device))

    # Fill remaining seats with baseline players if < 4 agents
    num_agents = len(agents)
    baseline_types = []
    if num_agents < 4:
        baseline_types = ["Random", "Heuristic", "Random", "Heuristic"]
        baseline_types = baseline_types[:(4 - num_agents)]
        print(f"\nFilling remaining seats with: {baseline_types}")

    # Run games with rotation
    print(f"\nRunning {num_games} games ({holes_per_game} holes each)...\n")

    all_results = []

    for game_num in range(num_games):
        # Determine rotation for this game
        if rotate_positions:
            rotation_offset = game_num % 4
        else:
            rotation_offset = 0

        # Build seat assignments: rotate DQN agents through all 4 seats
        agent_to_seat = {}
        seat_to_agent = {}
        player_names = []
        player_types = []

        # Assign DQN agents to rotated seats
        for agent_idx in range(num_agents):
            seat = (agent_idx + rotation_offset) % 4
            agent_to_seat[agent_idx] = seat
            seat_to_agent[seat] = agent_idx

        # Fill remaining seats with baselines
        baseline_iter = iter(baseline_types)
        for seat in range(4):
            if seat in seat_to_agent:
                agent_idx = seat_to_agent[seat]
                player_names.append(agent_names[agent_idx])
                player_types.append("OfflineDQN")
            else:
                baseline_type = next(baseline_iter)
                player_names.append(f"{baseline_type}_{seat}")
                player_types.append(baseline_type)
                seat_to_agent[seat] = None

        # Play each hole with FRESH player objects
        for hole in range(1, holes_per_game + 1):
            # Create fresh players for this hole
            players = []
            for seat in range(4):
                players.append(Player(
                    name=player_names[seat],
                    id=seat,
                    type=player_types[seat]
                ))

            golf = Golf(players=players, deck_type="French", verbose=False)

            # Map agents to correct seats for this game
            agent_array = [None, None, None, None]
            for agent_idx, seat in agent_to_seat.items():
                agent_array[seat] = agents[agent_idx]

            game_results, _ = play_game(
                golf,
                game_num=game_num,
                hole=hole,
                Q={},
                model=None,
                rank_cutoff=4,
                verbose=False,
                shuffle=True,
                transition_logger=None,
                offline_agent=agent_array,
            )

            # Add agent identity to results
            for result in game_results:
                seat = result["player_id"]
                agent_idx = seat_to_agent[seat]
                if agent_idx is not None:
                    result["agent_name"] = agent_names[agent_idx]
                    result["agent_type"] = "DQN"
                else:
                    result["agent_name"] = player_names[seat]
                    result["agent_type"] = player_types[seat]
                result["rotation"] = rotation_offset

            all_results.extend(game_results)

        if (game_num + 1) % 10 == 0 or game_num == 0:
            print(f"  Completed {game_num + 1}/{num_games} games")

    # Convert to DataFrame and calculate ranks
    df = pd.DataFrame(all_results)
    df["rank"] = df.groupby(["game", "hole"])["score"].rank(method="min").astype(int)

    # Compute statistics per agent
    print("\n" + "="*80)
    print("SELF-PLAY EVALUATION RESULTS")
    print("="*80 + "\n")

    agent_stats = []
    for agent_name in agent_names:
        agent_df = df[df["agent_name"] == agent_name]

        if len(agent_df) == 0:
            continue

        wins = (agent_df["rank"] == 1).sum()
        total_games = len(agent_df)
        win_rate = wins / total_games if total_games > 0 else 0.0

        stats = {
            "agent_name": agent_name,
            "games": total_games,
            "wins": int(wins),
            "win_rate": win_rate,
            "score_mean": float(agent_df["score"].mean()),
            "score_median": float(agent_df["score"].median()),
            "score_std": float(agent_df["score"].std()),
            "rank_mean": float(agent_df["rank"].mean()),
            "rank_median": float(agent_df["rank"].median()),
        }
        agent_stats.append(stats)

        print(f"{agent_name}:")
        print(f"  Win Rate:      {stats['win_rate']:.2%}")
        print(f"  Score (mean):  {stats['score_mean']:.2f} ± {stats['score_std']:.2f}")
        print(f"  Score (median): {stats['score_median']:.2f}")
        print(f"  Rank (mean):   {stats['rank_mean']:.2f}")
        print(f"  Rank (median): {stats['rank_median']:.1f}")
        print()

    # Add baseline stats if present
    baseline_stats = []
    for baseline_type in set(baseline_types):
        baseline_df = df[df["agent_type"] == baseline_type]
        if len(baseline_df) > 0:
            stats = {
                "agent_name": f"{baseline_type} (baseline)",
                "win_rate": float((baseline_df["rank"] == 1).sum() / len(baseline_df)),
                "score_mean": float(baseline_df["score"].mean()),
                "score_median": float(baseline_df["score"].median()),
            }
            baseline_stats.append(stats)
            print(f"{baseline_type} (baseline):")
            print(f"  Win Rate:      {stats['win_rate']:.2%}")
            print(f"  Score (mean):  {stats['score_mean']:.2f}")
            print(f"  Score (median): {stats['score_median']:.2f}")
            print()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "self_play_results.csv"
    df.to_csv(results_file, index=False)

    summary_file = output_dir / "self_play_summary.json"
    with summary_file.open("w") as f:
        json.dump({
            "agent_stats": agent_stats,
            "baseline_stats": baseline_stats,
            "num_games": num_games,
            "holes_per_game": holes_per_game,
            "rotation_enabled": rotate_positions,
        }, f, indent=2)

    print(f"Results saved to: {output_dir}")

    return {
        "agent_stats": agent_stats,
        "baseline_stats": baseline_stats,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN agents in self-play with rotation."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to checkpoints (1-4 agents)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names for each agent (defaults to exp_000, exp_001, ...)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--holes",
        type=int,
        default=9,
        help="Holes per game",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Disable player position rotation between games",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/self_play_eval"),
        help="Directory to save results",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Generate default names if not provided
    if args.names:
        if len(args.names) != len(args.checkpoints):
            print(f"Error: Number of names ({len(args.names)}) must match checkpoints ({len(args.checkpoints)})")
            return
        agent_names = args.names
    else:
        agent_names = [f"Agent_{i}" for i in range(len(args.checkpoints))]

    evaluate_self_play(
        checkpoints=args.checkpoints,
        agent_names=agent_names,
        num_games=args.games,
        holes_per_game=args.holes,
        rotate_positions=not args.no_rotation,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

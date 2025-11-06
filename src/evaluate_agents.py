"""Automated evaluation of trained DQN agents against baseline players."""

from __future__ import annotations

import argparse
import json
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from .simulation import SimulationConfig, run_simulation


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation."""
    checkpoint_path: Path
    experiment_name: str
    num_games: int = 100
    holes_per_game: int = 9
    dqn_player_id: int = 0
    device: str = "auto"
    seed: Optional[int] = None
    output_dir: Optional[Path] = None


def evaluate_agent(
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """Evaluate a single DQN agent against baseline players.

    Args:
        config: Evaluation configuration

    Returns:
        Dict with evaluation metrics
    """
    print(f"\nEvaluating: {config.experiment_name}")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"  Games: {config.num_games} × {config.holes_per_game} holes")
    print(f"  DQN player seat: {config.dqn_player_id}")

    # Create simulation config with DQN agent
    sim_config = SimulationConfig(
        num_games=config.num_games,
        holes_per_game=config.holes_per_game,
        verbose=False,
        shuffle=True,
        log_tensors=False,
        dqn_player_id=config.dqn_player_id,
        dqn_checkpoint=str(config.checkpoint_path),
        dqn_device=config.device,
    )

    # Run simulation
    results = run_simulation(config=sim_config, worker_id=None)

    # Parse results into DataFrame
    df = pd.DataFrame(results)

    # Compute per-player statistics
    player_stats = []
    for player_id in range(4):  # 4 players
        player_df = df[df["player_id"] == player_id]

        wins = (player_df["rank"] == 1).sum()
        total_games = len(player_df)
        win_rate = wins / total_games if total_games > 0 else 0.0

        stats = {
            "player_id": player_id,
            "player_type": player_df["player_type"].iloc[0] if len(player_df) > 0 else "Unknown",
            "games": total_games,
            "wins": int(wins),
            "win_rate": win_rate,
            "score_mean": float(player_df["score"].mean()),
            "score_std": float(player_df["score"].std()),
            "score_median": float(player_df["score"].median()),
            "rank_mean": float(player_df["rank"].mean()),
        }
        player_stats.append(stats)

    # Identify DQN agent stats
    dqn_stats = player_stats[config.dqn_player_id]

    # Compute opponent aggregate stats
    opponent_scores = []
    opponent_ranks = []
    for player_id, stats in enumerate(player_stats):
        if player_id != config.dqn_player_id:
            player_df = df[df["player_id"] == player_id]
            opponent_scores.extend(player_df["score"].tolist())
            opponent_ranks.extend(player_df["rank"].tolist())

    opponent_stats = {
        "score_mean": float(np.mean(opponent_scores)),
        "score_std": float(np.std(opponent_scores)),
        "rank_mean": float(np.mean(opponent_ranks)),
    }

    # Compile evaluation summary
    summary = {
        "experiment_name": config.experiment_name,
        "checkpoint": str(config.checkpoint_path),
        "num_games": config.num_games,
        "holes_per_game": config.holes_per_game,
        "dqn_player_id": config.dqn_player_id,
        "dqn_stats": dqn_stats,
        "opponent_stats": opponent_stats,
        "all_player_stats": player_stats,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {config.experiment_name}")
    print(f"{'='*60}")
    print(f"DQN Agent ({dqn_stats['player_type']}):")
    print(f"  Win rate: {dqn_stats['win_rate']:.1%}")
    print(f"  Avg score: {dqn_stats['score_mean']:.2f} ± {dqn_stats['score_std']:.2f}")
    print(f"  Avg rank: {dqn_stats['rank_mean']:.2f}")
    print(f"\nOpponents (aggregated):")
    print(f"  Avg score: {opponent_stats['score_mean']:.2f} ± {opponent_stats['score_std']:.2f}")
    print(f"  Avg rank: {opponent_stats['rank_mean']:.2f}")

    # Save detailed results if output dir provided
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save game-by-game results
        results_csv = config.output_dir / f"{config.experiment_name}_games.csv"
        df.to_csv(results_csv, index=False)

        # Save summary
        summary_json = config.output_dir / f"{config.experiment_name}_summary.json"
        with summary_json.open("w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved results to: {config.output_dir}")

    return summary


def evaluate_multiple_agents(
    checkpoint_dirs: List[Path],
    num_games: int = 100,
    holes_per_game: int = 9,
    dqn_player_id: int = 0,
    device: str = "auto",
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate multiple trained agents and compare results.

    Args:
        checkpoint_dirs: List of directories containing trained checkpoints
        num_games: Number of games to evaluate
        holes_per_game: Holes per game
        dqn_player_id: Which player seat the DQN controls
        device: Device for inference
        output_dir: Directory to save results
        seed: Random seed for evaluation

    Returns:
        List of evaluation summaries
    """
    results = []

    print(f"\n{'#'*80}")
    print(f"EVALUATING {len(checkpoint_dirs)} AGENTS")
    print(f"{'#'*80}\n")

    for i, checkpoint_dir in enumerate(checkpoint_dirs, 1):
        checkpoint_path = checkpoint_dir / "offline_dqn.pt"

        if not checkpoint_path.exists():
            print(f"[{i}/{len(checkpoint_dirs)}] SKIP: Checkpoint not found at {checkpoint_path}")
            continue

        experiment_name = checkpoint_dir.name

        eval_config = EvaluationConfig(
            checkpoint_path=checkpoint_path,
            experiment_name=experiment_name,
            num_games=num_games,
            holes_per_game=holes_per_game,
            dqn_player_id=dqn_player_id,
            device=device,
            seed=seed,
            output_dir=output_dir,
        )

        print(f"\n[{i}/{len(checkpoint_dirs)}] Evaluating: {experiment_name}")

        try:
            summary = evaluate_agent(eval_config)
            results.append(summary)
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {experiment_name}: {e}")
            results.append({
                "experiment_name": experiment_name,
                "checkpoint": str(checkpoint_path),
                "status": "failed",
                "error": str(e),
            })

    return results


def compare_agents(
    evaluation_results: List[Dict[str, Any]],
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Create a comparison table of agent performance.

    Args:
        evaluation_results: List of evaluation summaries
        output_file: Optional path to save comparison CSV

    Returns:
        DataFrame with comparison metrics
    """
    rows = []

    for result in evaluation_results:
        if result.get("status") == "failed":
            continue

        dqn_stats = result.get("dqn_stats", {})
        opponent_stats = result.get("opponent_stats", {})

        row = {
            "experiment": result["experiment_name"],
            "win_rate": dqn_stats.get("win_rate", 0.0),
            "dqn_score_mean": dqn_stats.get("score_mean", 0.0),
            "dqn_score_std": dqn_stats.get("score_std", 0.0),
            "dqn_rank_mean": dqn_stats.get("rank_mean", 0.0),
            "opponent_score_mean": opponent_stats.get("score_mean", 0.0),
            "opponent_rank_mean": opponent_stats.get("rank_mean", 0.0),
            "score_delta": dqn_stats.get("score_mean", 0.0) - opponent_stats.get("score_mean", 0.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Sort by win rate (descending) and score delta (ascending, since lower is better in golf)
        df = df.sort_values(["win_rate", "score_delta"], ascending=[False, True])

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, float_format="%.4f")
        print(f"\nSaved comparison to: {output_file}")

    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN agents against baseline players."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("tmp/experiments"),
        help="Directory containing experiment subdirectories with checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Single checkpoint to evaluate (instead of scanning experiments-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/evaluations"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to evaluate",
    )
    parser.add_argument(
        "--holes",
        type=int,
        default=9,
        help="Holes per game",
    )
    parser.add_argument(
        "--dqn-player-id",
        type=int,
        default=0,
        help="Which player seat (0-3) the DQN controls",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for DQN inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for evaluation",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Determine which checkpoints to evaluate
    if args.checkpoint:
        # Single checkpoint mode
        experiment_name = args.checkpoint.parent.name
        config = EvaluationConfig(
            checkpoint_path=args.checkpoint,
            experiment_name=experiment_name,
            num_games=args.games,
            holes_per_game=args.holes,
            dqn_player_id=args.dqn_player_id,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        results = [evaluate_agent(config)]
    else:
        # Multi-experiment mode
        if not args.experiments_dir.exists():
            print(f"Error: Experiments directory not found: {args.experiments_dir}")
            return

        # Find all experiment directories with checkpoints
        checkpoint_dirs = []
        for subdir in sorted(args.experiments_dir.iterdir()):
            if subdir.is_dir() and (subdir / "offline_dqn.pt").exists():
                checkpoint_dirs.append(subdir)

        if not checkpoint_dirs:
            print(f"No checkpoints found in: {args.experiments_dir}")
            return

        print(f"Found {len(checkpoint_dirs)} experiments with checkpoints")

        results = evaluate_multiple_agents(
            checkpoint_dirs=checkpoint_dirs,
            num_games=args.games,
            holes_per_game=args.holes,
            dqn_player_id=args.dqn_player_id,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    # Create comparison table
    if len(results) > 1:
        print(f"\n{'#'*80}")
        print("AGENT COMPARISON")
        print(f"{'#'*80}\n")

        comparison_file = args.output_dir / "agent_comparison.csv"
        comparison_df = compare_agents(results, output_file=comparison_file)

        print(comparison_df.to_string(index=False))

        # Highlight top performers
        if len(comparison_df) > 0:
            print(f"\n{'='*60}")
            print("TOP PERFORMERS")
            print(f"{'='*60}")
            top_3 = comparison_df.head(3)
            for i, row in enumerate(top_3.itertuples(), 1):
                print(f"\n{i}. {row.experiment}")
                print(f"   Win rate: {row.win_rate:.1%}")
                print(f"   DQN score: {row.dqn_score_mean:.2f} ± {row.dqn_score_std:.2f}")
                print(f"   Score delta: {row.score_delta:+.2f} (vs opponents)")

    # Save aggregate results
    aggregate_file = args.output_dir / "evaluation_results.json"
    with aggregate_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

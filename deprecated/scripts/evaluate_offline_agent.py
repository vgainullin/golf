"""DEPRECATED: Use scripts/eval_hof.py or scripts/eval_vs_random.py instead.

This script uses the old non-vectorized simulation loop (src.simulation).
The newer scripts use the vectorized engine (src.vectorized_golf) and support
v3 models via the tournament infrastructure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.offline_agent import OfflineDQAgent
from src.simulation import (
    Player,
    Golf,
    play_game,
)


def build_roster(agent_position: int) -> List[Player]:
    roster: List[Player] = []
    seat_types = ["Random", "Heuristic", "Random"]
    seat_iter = iter(seat_types)
    for seat in range(4):
        if seat == agent_position:
            roster.append(Player(name="OfflineDQN", id=seat, type="OfflineDQN"))
        else:
            roster.append(Player(name=f"Opponent{seat}", id=seat, type=next(seat_iter)))
    return roster


def evaluate_agent(
    checkpoint: Path,
    *,
    games: int,
    holes: int,
    agent_position: int,
    output_dir: Path,
    device: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    agent = OfflineDQAgent(checkpoint, device=device)

    records: List[Dict[str, float]] = []
    q_table: Dict[str, Dict[str, float]] = {}  # Placeholder for play_game signature

    for game_idx in range(games):
        for hole in range(1, holes + 1):
            players = build_roster(agent_position)
            golf = Golf(players=players, deck_type="French", verbose=False)
            game_results, _ = play_game(
                golf,
                game_idx,
                hole,
                q_table,
                model=None,
                rank_cutoff=4,
                verbose=False,
                shuffle=True,
                transition_logger=None,
                offline_agent=agent,
            )
            player_types = {player.id: player.type for player in players}
            for item in game_results:
                enriched = dict(item)
                enriched["player_type"] = player_types[enriched["player_id"]]
                enriched["agent_controlled"] = enriched["player_type"] == "OfflineDQN"
                records.append(enriched)

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "offline_dqn_evaluation.csv", index=False)

    summary = {
        "games": games,
        "holes_per_game": holes,
        "agent_position": agent_position,
    }

    agent_scores = df[df["agent_controlled"]]["score"]
    summary["agent_score_mean"] = float(agent_scores.mean())
    summary["agent_score_std"] = float(agent_scores.std(ddof=0))

    opponent_scores = df[~df["agent_controlled"]]
    summary["opponent_score_mean"] = float(opponent_scores["score"].mean())
    summary["opponent_score_std"] = float(opponent_scores["score"].std(ddof=0))

    wins = (
        df.groupby(["game", "hole"])["score"]
        .transform(lambda s: s == s.min())
        & df["agent_controlled"]
    )
    summary["agent_win_rate"] = float(wins.mean())

    with (output_dir / "offline_dqn_evaluation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Golf games with the offline DQN agent against baseline opponents."
    )
    parser.add_argument(
        "--checkpoint",
        default="tmp/offline_dqn_attempt/offline_dqn.pt",
        help="Path to the offline DQN checkpoint.",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games to play.")
    parser.add_argument("--holes", type=int, default=9, help="Holes per game.")
    parser.add_argument(
        "--agent-position",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Seat index (0-3) where the DQN agent will play.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/offline_dqn_eval",
        help="Directory to store evaluation logs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the DQN agent on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, summary = evaluate_agent(
        Path(args.checkpoint),
        games=args.games,
        holes=args.holes,
        agent_position=args.agent_position,
        output_dir=Path(args.output_dir),
        device=args.device,
    )
    print("Evaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"Detailed results written to {args.output_dir}/offline_dqn_evaluation.csv")


if __name__ == "__main__":
    main()

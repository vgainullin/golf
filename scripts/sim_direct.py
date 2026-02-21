"""Simulate 100 nine-hole Golf games directly via simulation.py.

Uses the same game engine (Golf, play_game) as the original codebase,
with 4 Heuristic players rotated exactly like the API simulation:
  - Starting player rotates each hole within a game.
  - Game starting offset rotates across games.

This serves as the ground-truth baseline to compare against the API sim.

Usage:
    python scripts/sim_direct.py
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List

from src.simulation import Golf, GolfDeck, Player, play_game

NUM_PLAYERS = 4
HOLES_PER_GAME = 9
NUM_GAMES = 100


def make_players() -> list[Player]:
    """Create 4 Heuristic players matching the API sim."""
    return [Player(name=f"PL{i}", id=i, type="Heuristic") for i in range(NUM_PLAYERS)]


def play_hole_direct(starting_player_id: int) -> dict[int, float]:
    """Play one hole using simulation.py's play_game and return {pid: score}.

    To honour starting_player_id we rotate the player list so that the
    desired starter is at index 0, then map results back to original IDs.
    """
    # Build players in rotated order so play_game (which always iterates
    # 0..N-1) effectively starts with starting_player_id.
    base_players = make_players()
    rotated = base_players[starting_player_id:] + base_players[:starting_player_id]
    # Reassign sequential ids so Golf iterates them 0..N-1
    for i, p in enumerate(rotated):
        p.id = i

    golf = Golf(players=rotated, deck_type="French", verbose=False)
    Q: Dict[str, Dict[str, float]] = {}

    results, _ = play_game(
        golf,
        game_num=0,
        hole=0,
        Q=Q,
        model=None,
        rank_cutoff=4,
        verbose=False,
        shuffle=True,
    )

    # Map back: rotated index -> original player id
    scores: dict[int, float] = {}
    for r in results:
        rotated_idx = r["player_id"]
        original_id = (rotated_idx + starting_player_id) % NUM_PLAYERS
        scores[original_id] = float(r["score"])
    return scores


def main() -> None:
    all_totals = {pid: 0.0 for pid in range(NUM_PLAYERS)}
    wins = {pid: 0 for pid in range(NUM_PLAYERS)}

    for game_num in range(NUM_GAMES):
        game_start = game_num % NUM_PLAYERS
        hole_accum = {pid: 0.0 for pid in range(NUM_PLAYERS)}

        for hole in range(HOLES_PER_GAME):
            starting = (game_start + hole) % NUM_PLAYERS
            scores = play_hole_direct(starting)
            for pid, sc in scores.items():
                hole_accum[pid] += sc

        game_winner = min(hole_accum, key=hole_accum.get)
        wins[game_winner] += 1
        for pid in range(NUM_PLAYERS):
            all_totals[pid] += hole_accum[pid]

        if (game_num + 1) % 25 == 0:
            print(f"  ... completed {game_num + 1}/{NUM_GAMES} games")

    print(f"\n{'='*60}")
    print(f"  DIRECT simulation.py: {NUM_GAMES} games x {HOLES_PER_GAME} holes")
    print(f"  (4 Heuristic players, rotating start)")
    print(f"{'='*60}")
    print(f"{'Player':<10} {'Avg 9-Hole':<14} {'Total':<10} {'Wins':<8} {'Win %':<8}")
    print("-" * 50)
    for pid in range(NUM_PLAYERS):
        avg = all_totals[pid] / NUM_GAMES
        print(
            f"Player {pid:<3} {avg:<14.1f} {all_totals[pid]:<10.0f} "
            f"{wins[pid]:<8} {wins[pid] / NUM_GAMES * 100:<7.1f}%"
        )

    overall = min(all_totals, key=all_totals.get)
    most_wins = max(wins, key=wins.get)
    print(f"\nLowest avg score:  Player {overall} ({all_totals[overall] / NUM_GAMES:.1f})")
    print(f"Most game wins:    Player {most_wins} ({wins[most_wins]}/{NUM_GAMES})")


if __name__ == "__main__":
    main()

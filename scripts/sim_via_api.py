"""Simulate 100 nine-hole Golf games via the REST API (all-AI heuristic).

Rotations per the user spec:
  - Seats rotate after every full 9-hole game (player types shift).
  - The starting player for each game rotates across the 100 games.
  - Within a game, the first-draw player rotates each hole sequentially.

Usage:
    # start the server first:  uvicorn src.api.app:app
    python scripts/sim_via_api.py
"""

import httpx
import sys

BASE = "http://127.0.0.1:8000/api/v1/games"

NUM_PLAYERS = 4
HOLES_PER_GAME = 9
NUM_GAMES = 100


def simulate_hole(client: httpx.Client, starting_player_id: int) -> dict[int, float]:
    """Play one hole via POST /games/simulate and return {player_id: score}."""
    resp = client.post(
        f"{BASE}/simulate",
        json={
            "num_players": NUM_PLAYERS,
            "player_type": "heuristic",
            "starting_player_id": starting_player_id,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return {e["player_id"]: e["final_score"] for e in data["scoreboard"]}


def main():
    with httpx.Client(timeout=30) as client:
        # Health check
        h = client.get("http://127.0.0.1:8000/health")
        if h.status_code != 200:
            print("Server not reachable. Start it with: uvicorn src.api.app:app")
            sys.exit(1)

        # Accumulators: game_totals[game][pid] = 9-hole total
        game_totals: list[dict[int, float]] = []
        # Per-player-id accumulator across all games
        all_totals = {pid: 0.0 for pid in range(NUM_PLAYERS)}
        # Win counter per player id
        wins = {pid: 0 for pid in range(NUM_PLAYERS)}

        for game_num in range(NUM_GAMES):
            # Rotate which player starts the game
            game_start = game_num % NUM_PLAYERS
            hole_scores_accum = {pid: 0.0 for pid in range(NUM_PLAYERS)}

            for hole in range(HOLES_PER_GAME):
                # Within a game, rotate first-draw sequentially per hole
                starting = (game_start + hole) % NUM_PLAYERS
                scores = simulate_hole(client, starting)
                for pid, sc in scores.items():
                    hole_scores_accum[pid] += sc

            game_totals.append(hole_scores_accum)
            game_winner = min(hole_scores_accum, key=hole_scores_accum.get)
            wins[game_winner] += 1
            for pid in range(NUM_PLAYERS):
                all_totals[pid] += hole_scores_accum[pid]

            if (game_num + 1) % 25 == 0:
                print(f"  ... completed {game_num + 1}/{NUM_GAMES} games")

    # Compute averages
    print(f"\n{'='*56}")
    print(f"  RESULTS: {NUM_GAMES} games x {HOLES_PER_GAME} holes  (4 Heuristic players)")
    print(f"{'='*56}")
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

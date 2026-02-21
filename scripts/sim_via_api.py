"""Play a 9-hole Golf game through the REST API with 4 heuristic players.

Player 0 is the 'human' seat controlled by this script using simple heuristic
logic (take discard if rank < 4, else draw; place if it lowers score, else
discard+flip).  Players 1-3 are server-side heuristics.

Usage:
    python scripts/sim_via_api.py
"""

import httpx
import sys

BASE = "http://127.0.0.1:8000/api/v1/games"

SCORE_MAP = {
    "2": -2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "X": 10, "J": 10, "Q": 10, "K": 0, "A": 1,
}

HOLES = 9


def rank_score(rank: str) -> int:
    return SCORE_MAP.get(rank, 10)


def pick_draw_action(state: dict) -> str:
    """Heuristic: take discard if its rank scores < 4, else draw from deck."""
    fc = state.get("face_card")
    if fc and rank_score(fc["rank"]) < 4:
        return "take_discard"
    return "draw_deck"


def pick_place_action(state: dict) -> dict:
    """Heuristic: place the held card at the position that helps most,
    or discard+flip if the card is bad."""
    holding = state.get("holding")
    if not holding:
        return {"action": "discard", "position": 0}

    held_score = rank_score(holding["rank"])
    cards = state["players"][0]["cards"]  # human is player 0

    # Find the best face-down slot to flip (for discard path)
    face_down = []
    for r in range(2):
        for c in range(3):
            pos = r * 3 + c
            if cards[r][c] == "?":
                face_down.append(pos)

    # Find the worst visible card we could replace
    worst_pos = None
    worst_score = -99
    for r in range(2):
        for c in range(3):
            pos = r * 3 + c
            cell = cards[r][c]
            if cell == "?":
                continue
            cell_rank = cell[0]
            s = rank_score(cell_rank)
            if s > worst_score:
                worst_score = s
                worst_pos = pos

    # Place the card if it improves on the worst visible card
    if worst_pos is not None and held_score < worst_score:
        return {"action": "place", "position": worst_pos}

    # Otherwise place into a face-down slot if the card is decent
    if held_score <= 4 and face_down:
        return {"action": "place", "position": face_down[0]}

    # Discard and flip a face-down card
    if face_down:
        return {"action": "discard", "position": face_down[0]}

    # All revealed – just place somewhere
    return {"action": "place", "position": worst_pos or 0}


def play_one_hole(client: httpx.Client, hole_num: int) -> dict:
    """Play a single hole and return {player_id: score}."""
    resp = client.post(BASE, json={
        "num_players": 4,
        "human_player_id": 0,
        "opponent_type": "heuristic",
    })
    resp.raise_for_status()
    data = resp.json()
    game_id = data["game_id"]
    state = data["state"]
    print(f"\n--- Hole {hole_num} (game {game_id}) ---")

    turn = 0
    while not state["game_over"]:
        # Draw
        action = pick_draw_action(state)
        r = client.post(f"{BASE}/{game_id}/draw", json={"action": action})
        r.raise_for_status()
        state = r.json()["state"]

        if state["game_over"]:
            break

        # Place
        body = pick_place_action(state)
        r = client.post(f"{BASE}/{game_id}/place", json=body)
        r.raise_for_status()
        result = r.json()
        state = result["state"]

        turn += 1
        if result["ai_actions"]:
            for a in result["ai_actions"]:
                print(f"  {a}")

    # Fetch scores
    r = client.get(f"{BASE}/{game_id}/scores")
    r.raise_for_status()
    scores_data = r.json()
    hole_scores = {}
    for entry in scores_data["scoreboard"]:
        pid = entry["player_id"]
        sc = entry["final_score"]
        hole_scores[pid] = sc
    winner = scores_data["winner"]
    print(f"  Turns: {turn}  |  Scores: {hole_scores}  |  Hole winner: Player {winner}")

    # Clean up
    client.delete(f"{BASE}/{game_id}")
    return hole_scores


def main():
    totals = {i: 0.0 for i in range(4)}
    with httpx.Client(timeout=30) as client:
        # Quick health check
        h = client.get("http://127.0.0.1:8000/health")
        if h.status_code != 200:
            print("Server not reachable. Start it with: uvicorn src.api.app:app")
            sys.exit(1)

        for hole in range(1, HOLES + 1):
            hole_scores = play_one_hole(client, hole)
            for pid, sc in hole_scores.items():
                totals[pid] += sc

    print("\n========== FINAL RESULTS (9 holes) ==========")
    print(f"{'Player':<12} {'Type':<12} {'Total Score':<12}")
    print("-" * 36)
    for pid in sorted(totals):
        label = "Human-Heur" if pid == 0 else "Heuristic"
        print(f"Player {pid:<5} {label:<12} {totals[pid]:<12.0f}")
    overall_winner = min(totals, key=totals.get)
    print(f"\nOverall winner: Player {overall_winner} with {totals[overall_winner]:.0f}")


if __name__ == "__main__":
    main()

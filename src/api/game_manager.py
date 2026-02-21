"""In-memory game session manager.

Holds active Golf game instances keyed by a unique game ID.  Each session
tracks the game object plus bookkeeping (whose turn, what stage, etc.)
so the REST layer stays thin.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.simulation import (
    Card,
    Golf,
    GolfDeck,
    Player,
    calc_opt_heuristic_position,
    encode_pos_tuple,
    get_player_action,
)


@dataclass
class GameSession:
    game_id: str
    golf: Golf
    human_player_id: int
    hole: int = 1
    round_num: int = 0
    current_player_id: int = 0
    stage: int = 0  # 0 = draw, 1 = place/discard
    ai_log: list[str] = field(default_factory=list)


# Module-level store (simple dict; swap for Redis/DB later if needed)
_sessions: dict[str, GameSession] = {}


def _card_display(card: Any) -> str:
    """Return a display string for a card or '?' for unknowns."""
    if card is None or card == "?":
        return "?"
    if hasattr(card, "rank") and hasattr(card, "suit"):
        return f"{card.rank}{card.suit[0]}"
    return str(card)


def _card_out(card: Any) -> dict | None:
    """Convert a Card namedtuple to a dict suitable for CardOut, or None."""
    if card is None or card == "?":
        return None
    return {"rank": card.rank, "suit": card.suit}


def create_game(
    num_players: int = 4,
    human_player_id: int = 0,
    opponent_type: str = "heuristic",
) -> GameSession:
    """Create a new Golf game, shuffle, deal, and return the session."""
    type_map = {
        "human": "Human",
        "heuristic": "Heuristic",
        "random": "Random",
    }
    ai_type = type_map.get(opponent_type, "Heuristic")

    players = []
    for i in range(num_players):
        if i == human_player_id:
            p = Player(name=f"Player_{i}", id=i, type="Human")
        else:
            p = Player(name=f"Player_{i}", id=i, type=ai_type)
        players.append(p)

    golf = Golf(players=players, verbose=False)
    golf.shuffle()
    golf.deal()

    # Flip two initial cards for every player (standard Golf opening)
    import random as _rand
    for player in golf.players:
        positions = [(r, c) for r in range(2) for c in range(3)]
        chosen = _rand.sample(positions, 2)
        for r, c in chosen:
            player.open_cards[r][c] = player.cards[r][c]
        player.calculate_score()

    game_id = uuid.uuid4().hex[:12]
    session = GameSession(
        game_id=game_id,
        golf=golf,
        human_player_id=human_player_id,
        current_player_id=0,
    )
    _sessions[game_id] = session
    return session


def get_session(game_id: str) -> GameSession | None:
    return _sessions.get(game_id)


def delete_session(game_id: str) -> bool:
    return _sessions.pop(game_id, None) is not None


def build_game_state(session: GameSession) -> dict:
    """Serialize current game state into a dict matching GameState schema."""
    golf = session.golf
    human_id = session.human_player_id

    players_out = []
    for p in golf.players:
        # For the human player show their open_cards view; for others show open_cards too
        display = []
        for row in range(2):
            row_cards = []
            for col in range(3):
                row_cards.append(_card_display(p.open_cards[row][col]))
            display.append(row_cards)
        players_out.append({
            "player_id": p.id,
            "player_type": p.type.lower(),
            "cards": display,
            "score": float(p.score),
            "is_current": p.id == session.current_player_id,
        })

    holding = None
    human = golf.players[human_id]
    if session.current_player_id == human_id and human.holding:
        holding = _card_out(human.holding)

    return {
        "game_id": session.game_id,
        "hole": session.hole,
        "round": session.round_num,
        "current_player_id": session.current_player_id,
        "stage": session.stage,
        "face_card": _card_out(golf.face_card),
        "holding": holding,
        "players": players_out,
        "game_over": golf.game_over,
        "deck_remaining": len(golf.deck),
        "message": "",
    }


def _play_ai_turn(session: GameSession, player_id: int) -> str:
    """Execute a full AI turn (stage 0 + stage 1) and return a summary string."""
    golf = session.golf
    player = golf.players[player_id]
    take_random = player.type == "Random"

    # Stage 0: draw
    action0, pos0, _ = get_player_action(
        deepcopy(golf), player_id, 0, rank_cutoff=4, take_random_action=take_random,
    )
    golf.take_action(player_id, [0, action0, pos0])
    player.gather_game_state(golf)

    draw_desc = "took discard" if action0 == 0 else "drew from deck"

    # Stage 1: place/discard
    action1, pos1, _ = get_player_action(
        deepcopy(golf), player_id, 1, rank_cutoff=4, take_random_action=take_random,
    )
    golf.take_action(player_id, [1, action1, pos1])
    player.gather_game_state(golf)
    player.calculate_score()

    if action1 == 0 and pos1 is not None:
        row, col = divmod(pos1, 3)
        place_desc = f"placed at ({row},{col})"
    else:
        if pos1 is not None:
            row, col = divmod(pos1, 3)
            place_desc = f"discarded & flipped ({row},{col})"
        else:
            place_desc = "discarded"

    return f"Player {player_id} ({player.type}): {draw_desc}, {place_desc}"


def execute_draw(session: GameSession, action: str) -> float:
    """Execute the human player's stage-0 (draw) action. Returns reward."""
    golf = session.golf
    pid = session.human_player_id

    if action == "take_discard":
        action_code = 0  # take face card
    else:
        action_code = 1  # draw from deck

    reward = golf.take_action(pid, [0, action_code, None])
    golf.players[pid].gather_game_state(golf)
    session.stage = 1
    return float(reward)


def execute_place(session: GameSession, action: str, position: int) -> float:
    """Execute the human player's stage-1 (place/discard) action. Returns reward."""
    golf = session.golf
    pid = session.human_player_id

    if action == "place":
        action_code = 0
    else:
        action_code = 1  # discard + flip

    reward = golf.take_action(pid, [1, action_code, position])
    golf.players[pid].gather_game_state(golf)
    golf.players[pid].calculate_score()
    session.stage = 0
    return float(reward)


def advance_to_human_turn(session: GameSession) -> list[str]:
    """After the human finishes, play all AI turns until it's the human's turn again.

    Also checks end-of-round conditions.  Returns list of AI action summaries.
    """
    golf = session.golf
    ai_summaries: list[str] = []

    # Move to next player
    session.current_player_id = (session.current_player_id + 1) % golf.num_players

    while session.current_player_id != session.human_player_id:
        if golf.game_over:
            break

        pid = session.current_player_id

        # Check if this player triggers last turn
        if golf.last_turn and golf.end_game_player_id == pid:
            golf.game_over = True
            break

        summary = _play_ai_turn(session, pid)
        ai_summaries.append(summary)

        # Check if all cards are revealed -> last turn
        player = golf.players[pid]
        player.gather_game_state(golf)
        if "?" not in player.game_state:
            golf.last_turn = True
            golf.end_game_player_id = pid

        # Check deck depletion
        if len(golf.deck) < golf.num_players + 2:
            golf.deck = GolfDeck()
            golf.shuffle()

        session.current_player_id = (session.current_player_id + 1) % golf.num_players

    # If back to human, check their end conditions too
    if not golf.game_over:
        human = golf.players[session.human_player_id]
        human.gather_game_state(golf)
        if golf.last_turn and golf.end_game_player_id == session.human_player_id:
            golf.game_over = True

    session.round_num += 1
    return ai_summaries


def get_final_scores(session: GameSession) -> list[dict]:
    """Compute final scores for all players."""
    golf = session.golf
    results = []
    for p in golf.players:
        p.calculate_score(final=True)
        results.append({
            "player_id": p.id,
            "player_type": p.type.lower(),
            "final_score": float(p.score),
        })
    return results

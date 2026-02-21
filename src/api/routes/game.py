"""REST endpoints for playing a game of Golf."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.game_manager import (
    advance_to_human_turn,
    build_game_state,
    create_game,
    delete_session,
    execute_draw,
    execute_place,
    get_final_scores,
    get_session,
)
from src.api.game_models import (
    ActionResponse,
    CreateGameRequest,
    CreateGameResponse,
    DrawActionRequest,
    GameOverResponse,
    GameState,
    PlaceActionRequest,
)

router = APIRouter(prefix="/games", tags=["game"])

_RULES = {
    "goal": "Lowest score wins.",
    "setup": "Each player gets 6 cards in a 2x3 grid (face-down). Two are flipped to start.",
    "turn": [
        "1. Draw: take the face-up discard OR draw from the deck.",
        "2. Then either PLACE the drawn card into your grid (replacing a card) "
        "or DISCARD it and flip one of your face-down cards.",
    ],
    "end": "The round ends once any player has all 6 cards face-up. Everyone else gets one final turn.",
    "scoring": {
        "2": -2,
        "3-9": "face value",
        "10/J/Q": 10,
        "K": 0,
        "A": 1,
        "column_pair": "Two matching cards in the same column score 0 instead.",
    },
    "grid_positions": "Row-major: [0,1,2] = top row, [3,4,5] = bottom row.",
}


@router.get("/rules")
async def rules() -> dict:
    """Return a concise summary of the Golf card game rules."""
    return _RULES


def _require_session(game_id: str):
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    return session


def _require_not_over(session):
    if session.golf.game_over:
        raise HTTPException(status_code=400, detail="Game is already over")


def _require_human_turn(session):
    if session.current_player_id != session.human_player_id:
        raise HTTPException(status_code=400, detail="It is not the human player's turn")


# ---------------------------------------------------------------------------
# POST /games  – create a new game
# ---------------------------------------------------------------------------


@router.post("", response_model=CreateGameResponse, status_code=201)
async def new_game(body: CreateGameRequest | None = None) -> CreateGameResponse:
    """Start a new Golf card game session."""
    body = body or CreateGameRequest()
    if body.human_player_id >= body.num_players:
        raise HTTPException(
            status_code=422,
            detail="human_player_id must be less than num_players",
        )
    session = create_game(
        num_players=body.num_players,
        human_player_id=body.human_player_id,
        opponent_type=body.opponent_type.value,
    )
    # If human is not player 0, play AI turns up to the human
    if session.human_player_id != 0:
        ai_msgs = advance_to_human_turn(session)
        # reset round since this is the opening
        session.round_num = 0

    state = build_game_state(session)
    state["message"] = "Game started. You must draw a card."
    return CreateGameResponse(game_id=session.game_id, state=GameState(**state))


# ---------------------------------------------------------------------------
# GET /games/{game_id}  – get current state
# ---------------------------------------------------------------------------


@router.get("/{game_id}", response_model=GameState)
async def get_game_state(game_id: str) -> GameState:
    """Get the current state of a game."""
    session = _require_session(game_id)
    state = build_game_state(session)
    if session.golf.game_over:
        state["message"] = "Game over! Fetch /scores for final results."
    elif session.stage == 0:
        state["message"] = "Your turn: draw a card (draw_deck or take_discard)."
    else:
        state["message"] = "Your turn: place the card or discard and flip."
    return GameState(**state)


# ---------------------------------------------------------------------------
# POST /games/{game_id}/draw  – stage 0: draw a card
# ---------------------------------------------------------------------------


@router.post("/{game_id}/draw", response_model=ActionResponse)
async def draw_card(game_id: str, body: DrawActionRequest) -> ActionResponse:
    """Stage 0: Draw a card from the deck or take the top discard."""
    session = _require_session(game_id)
    _require_not_over(session)
    _require_human_turn(session)

    if session.stage != 0:
        raise HTTPException(
            status_code=400,
            detail="You already drew a card. Use /place to place or discard it.",
        )

    reward = execute_draw(session, body.action.value)
    state = build_game_state(session)
    state["message"] = "Card drawn. Now place it or discard and flip a face-down card."
    return ActionResponse(reward=reward, ai_actions=[], state=GameState(**state))


# ---------------------------------------------------------------------------
# POST /games/{game_id}/place  – stage 1: place or discard
# ---------------------------------------------------------------------------


@router.post("/{game_id}/place", response_model=ActionResponse)
async def place_card(game_id: str, body: PlaceActionRequest) -> ActionResponse:
    """Stage 1: Place the drawn card into your grid or discard it and flip a card."""
    session = _require_session(game_id)
    _require_not_over(session)
    _require_human_turn(session)

    if session.stage != 1:
        raise HTTPException(
            status_code=400,
            detail="You must draw a card first. Use /draw.",
        )

    golf = session.golf
    human = golf.players[session.human_player_id]

    # Validate position for discard action (must be face-down)
    if body.action.value == "discard":
        row, col = divmod(body.position, 3)
        if human.open_cards[row][col] != "?":
            raise HTTPException(
                status_code=400,
                detail=f"Position {body.position} is already face-up. Pick a face-down card to flip.",
            )

    reward = execute_place(session, body.action.value, body.position)

    # Check if human revealed all cards
    human.gather_game_state(golf)
    if "?" not in human.game_state:
        golf.last_turn = True
        golf.end_game_player_id = session.human_player_id

    # Now play all AI turns until it's the human's turn again
    ai_msgs = advance_to_human_turn(session)

    state = build_game_state(session)
    if golf.game_over:
        state["message"] = "Game over! Fetch /scores for final results."
    else:
        state["message"] = "Your turn: draw a card (draw_deck or take_discard)."

    return ActionResponse(reward=reward, ai_actions=ai_msgs, state=GameState(**state))


# ---------------------------------------------------------------------------
# GET /games/{game_id}/scores  – final scoreboard
# ---------------------------------------------------------------------------


@router.get("/{game_id}/scores", response_model=GameOverResponse)
async def get_scores(game_id: str) -> GameOverResponse:
    """Get final scores. Only available once the game is over."""
    session = _require_session(game_id)
    if not session.golf.game_over:
        raise HTTPException(status_code=400, detail="Game is not over yet")

    results = get_final_scores(session)
    winner = min(results, key=lambda r: r["final_score"])["player_id"]
    return GameOverResponse(
        game_id=game_id,
        scoreboard=results,
        winner=winner,
    )


# ---------------------------------------------------------------------------
# DELETE /games/{game_id}  – abandon a game
# ---------------------------------------------------------------------------


@router.delete("/{game_id}", status_code=204)
async def abandon_game(game_id: str) -> None:
    """Delete / abandon a game session."""
    if not delete_session(game_id):
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

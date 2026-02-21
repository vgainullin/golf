"""Pydantic models for the Golf card-game API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlayerType(str, Enum):
    human = "human"
    heuristic = "heuristic"
    random = "random"


class DrawAction(str, Enum):
    """Stage-0 action: where to draw a card from."""
    draw_deck = "draw_deck"
    take_discard = "take_discard"


class PlaceAction(str, Enum):
    """Stage-1 action: what to do with the drawn card."""
    place = "place"
    discard = "discard"


# ---------------------------------------------------------------------------
# Card representation
# ---------------------------------------------------------------------------


class CardOut(BaseModel):
    rank: str = Field(..., description="Card rank (2-9, X, J, Q, K, A)")
    suit: str = Field(..., description="Card suit (spades, diamonds, clubs, hearts)")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateGameRequest(BaseModel):
    num_players: int = Field(default=4, ge=2, le=4, description="Number of players (2-4)")
    human_player_id: Optional[int] = Field(default=0, ge=0, description="Which player the human controls (null for all-AI)")
    opponent_type: PlayerType = Field(
        default=PlayerType.heuristic,
        description="AI type for non-human players",
    )


class SimulateRequest(BaseModel):
    num_players: int = Field(default=4, ge=2, le=4, description="Number of players (2-4)")
    player_type: PlayerType = Field(default=PlayerType.heuristic, description="AI type for all players")
    starting_player_id: int = Field(default=0, ge=0, description="Which player draws first")


class SimulateResponse(BaseModel):
    game_id: str
    scoreboard: list[ScoreboardEntry]
    winner: int


class DrawActionRequest(BaseModel):
    action: DrawAction


class PlaceActionRequest(BaseModel):
    action: PlaceAction
    position: int = Field(
        ...,
        ge=0,
        le=5,
        description="Card position (0-5). Grid is row-major: [0,1,2] top row, [3,4,5] bottom row",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PlayerState(BaseModel):
    player_id: int
    player_type: str
    cards: list[list[str]] = Field(
        ...,
        description="2x3 grid of visible cards. '?' means face-down.",
    )
    score: float
    is_current: bool = False


class GameState(BaseModel):
    game_id: str
    hole: int
    round: int
    current_player_id: int
    stage: int = Field(..., description="0 = must draw, 1 = must place/discard")
    face_card: Optional[CardOut] = None
    holding: Optional[CardOut] = None
    players: list[PlayerState]
    game_over: bool
    deck_remaining: int
    message: str = ""


class CreateGameResponse(BaseModel):
    game_id: str
    state: GameState


class ActionResponse(BaseModel):
    reward: float
    ai_actions: list[str] = Field(
        default_factory=list,
        description="Summary of AI turns that happened after the human move",
    )
    state: GameState


class ScoreboardEntry(BaseModel):
    player_id: int
    player_type: str
    final_score: float


class GameOverResponse(BaseModel):
    game_id: str
    scoreboard: list[ScoreboardEntry]
    winner: int


class PlayerTableResponse(BaseModel):
    game_id: str
    player_id: int
    player_type: str
    cards: list[list[str]] = Field(
        ...,
        description="2x3 grid. Each cell is 'Rank+Suit' (e.g. 'Ks') or '?' if face-down.",
    )
    score: float
    is_current: bool

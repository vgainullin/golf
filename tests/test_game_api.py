"""Tests for the Golf card-game API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.game_manager import _sessions


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Ensure a clean session store for every test."""
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_rules(client: TestClient) -> None:
    resp = client.get("/api/v1/games/rules")
    assert resp.status_code == 200
    body = resp.json()
    assert "goal" in body
    assert "scoring" in body
    assert "turn" in body


# ---------------------------------------------------------------------------
# Create game
# ---------------------------------------------------------------------------


def test_create_game_defaults(client: TestClient) -> None:
    resp = client.post("/api/v1/games")
    assert resp.status_code == 201
    body = resp.json()
    assert "game_id" in body
    state = body["state"]
    assert state["game_over"] is False
    assert state["stage"] == 0
    assert len(state["players"]) == 4
    assert state["current_player_id"] == 0
    assert state["face_card"] is not None


def test_create_game_custom(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/games",
        json={"num_players": 2, "human_player_id": 0, "opponent_type": "random"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["state"]["players"]) == 2


def test_create_game_invalid_human_id(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/games",
        json={"num_players": 2, "human_player_id": 5},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get game state
# ---------------------------------------------------------------------------


def test_get_state(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    resp = client.get(f"/api/v1/games/{game_id}")
    assert resp.status_code == 200
    assert resp.json()["game_id"] == game_id


def test_get_state_not_found(client: TestClient) -> None:
    resp = client.get("/api/v1/games/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Draw action (stage 0)
# ---------------------------------------------------------------------------


def test_draw_from_deck(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    resp = client.post(
        f"/api/v1/games/{game_id}/draw",
        json={"action": "draw_deck"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"]["stage"] == 1
    assert body["state"]["holding"] is not None


def test_draw_take_discard(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    resp = client.post(
        f"/api/v1/games/{game_id}/draw",
        json={"action": "take_discard"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"]["stage"] == 1


def test_draw_wrong_stage(client: TestClient) -> None:
    """Drawing twice in a row should fail."""
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    client.post(f"/api/v1/games/{game_id}/draw", json={"action": "draw_deck"})
    resp = client.post(f"/api/v1/games/{game_id}/draw", json={"action": "draw_deck"})
    assert resp.status_code == 400
    assert "already drew" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Place action (stage 1)
# ---------------------------------------------------------------------------


def test_place_card(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    # Draw first
    client.post(f"/api/v1/games/{game_id}/draw", json={"action": "draw_deck"})

    # Place at position 0
    resp = client.post(
        f"/api/v1/games/{game_id}/place",
        json={"action": "place", "position": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    # After human places, AI plays, then it's human's turn again (stage 0)
    assert body["state"]["stage"] == 0


def test_place_wrong_stage(client: TestClient) -> None:
    """Placing without drawing first should fail."""
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    resp = client.post(
        f"/api/v1/games/{game_id}/place",
        json={"action": "place", "position": 0},
    )
    assert resp.status_code == 400
    assert "draw" in resp.json()["detail"].lower()


def test_discard_and_flip(client: TestClient) -> None:
    """Discard the drawn card and flip a face-down card."""
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]
    state = create.json()["state"]

    # Find a face-down position
    human_cards = state["players"][0]["cards"]
    face_down_pos = None
    for r in range(2):
        for c in range(3):
            if human_cards[r][c] == "?":
                face_down_pos = r * 3 + c
                break
        if face_down_pos is not None:
            break

    assert face_down_pos is not None, "Expected at least one face-down card"

    client.post(f"/api/v1/games/{game_id}/draw", json={"action": "draw_deck"})
    resp = client.post(
        f"/api/v1/games/{game_id}/place",
        json={"action": "discard", "position": face_down_pos},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Full turn cycle
# ---------------------------------------------------------------------------


def test_full_turn(client: TestClient) -> None:
    """Play a full turn: draw + place, verify AI actions are reported."""
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    client.post(f"/api/v1/games/{game_id}/draw", json={"action": "draw_deck"})
    resp = client.post(
        f"/api/v1/games/{game_id}/place",
        json={"action": "place", "position": 0},
    )
    body = resp.json()
    # AI actions should be reported
    assert isinstance(body["ai_actions"], list)
    # After the AI plays, it should be human's turn
    assert body["state"]["current_player_id"] == 0


# ---------------------------------------------------------------------------
# Scores endpoint
# ---------------------------------------------------------------------------


def test_scores_not_over(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    resp = client.get(f"/api/v1/games/{game_id}/scores")
    assert resp.status_code == 400
    assert "not over" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Delete game
# ---------------------------------------------------------------------------


def test_delete_game(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    resp = client.delete(f"/api/v1/games/{game_id}")
    assert resp.status_code == 204

    resp = client.get(f"/api/v1/games/{game_id}")
    assert resp.status_code == 404


def test_delete_not_found(client: TestClient) -> None:
    resp = client.delete("/api/v1/games/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Play a game to completion
# ---------------------------------------------------------------------------


def test_play_to_completion(client: TestClient) -> None:
    """Play turns until the game ends and verify the scoreboard."""
    create = client.post("/api/v1/games", json={"num_players": 2})
    game_id = create.json()["game_id"]

    for _ in range(50):  # upper bound on turns
        state_resp = client.get(f"/api/v1/games/{game_id}")
        state = state_resp.json()
        if state["game_over"]:
            break

        # Draw
        draw_resp = client.post(
            f"/api/v1/games/{game_id}/draw",
            json={"action": "draw_deck"},
        )
        if draw_resp.status_code != 200:
            break

        # Find a valid position for placing
        human_cards = draw_resp.json()["state"]["players"][0]["cards"]
        # Just place at position 0 always (replace whatever is there)
        place_resp = client.post(
            f"/api/v1/games/{game_id}/place",
            json={"action": "place", "position": 0},
        )
        if place_resp.status_code != 200:
            break
        if place_resp.json()["state"]["game_over"]:
            break

    # Game should be over
    scores_resp = client.get(f"/api/v1/games/{game_id}/scores")
    assert scores_resp.status_code == 200
    body = scores_resp.json()
    assert len(body["scoreboard"]) == 2
    assert "winner" in body

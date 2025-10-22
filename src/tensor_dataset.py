"""Utility helpers for working with tensor transition artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import json
import numpy as np

from .tensor_logger import decode_action_id


UNKNOWN_CARD_INDEX = 52


def _infer_num_players(state_tensor: np.ndarray) -> int:
    """Infer number of players from the tensor shape."""

    if state_tensor.ndim != 3:
        raise ValueError("State tensor must have shape (ranks, positions, suits).")

    _, pos_dim, _ = state_tensor.shape
    if pos_dim < 3 or (pos_dim - 3) % 7 != 0:
        raise ValueError(
            f"Position dimension {pos_dim} does not match Golf tensor layout."
        )
    return (pos_dim - 3) // 7


def _extract_card_index(column: np.ndarray, num_ranks: int) -> int:
    """Return the packed card index for a position column."""

    if column.ndim != 2:
        raise ValueError("Column slice must be 2D (ranks x suits).")
    present = np.argwhere(column)
    if present.size == 0:
        return UNKNOWN_CARD_INDEX
    # For deck/discard columns there may be multiple entries; choose the first.
    rank_idx, suit_idx = present[0]
    return int(suit_idx * num_ranks + rank_idx)


def _player_card_positions(player_id: int) -> Sequence[int]:
    base = player_id * 7
    return tuple(base + slot for slot in range(6))


def _player_holding_position(player_id: int) -> int:
    return player_id * 7 + 6


def _deck_position(num_players: int) -> int:
    return num_players * 7


def _discard_position(num_players: int) -> int:
    return num_players * 7 + 1


def _face_position(num_players: int) -> int:
    return num_players * 7 + 2


def tensor_to_player_tokens(
    state_tensor: np.ndarray,
    *,
    player_id: int,
    num_players: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    """Convert a full game tensor to QTransformer-style tokens.

    Returns a tuple of (player_cards[6], holding_card, discard_top)
    with values in [0, 52], where 52 represents "unknown".
    """

    num_ranks = state_tensor.shape[0]
    if num_players is None:
        num_players = _infer_num_players(state_tensor)

    player_cards: List[int] = []
    for pos in _player_card_positions(player_id):
        column = state_tensor[:, pos, :]
        player_cards.append(_extract_card_index(column, num_ranks))

    holding_column = state_tensor[:, _player_holding_position(player_id), :]
    holding_card = _extract_card_index(holding_column, num_ranks)

    face_column = state_tensor[:, _face_position(num_players), :]
    discard_top = _extract_card_index(face_column, num_ranks)

    return (
        np.asarray(player_cards, dtype=np.int64),
        int(holding_card),
        int(discard_top),
    )


@dataclass
class TensorTransitionRecord:
    index: int
    state: np.ndarray
    action_id: int
    reward: float
    next_state: np.ndarray
    done: bool
    metadata: Dict[str, object]

    def action_tuple(self) -> Tuple[int, Optional[int], Optional[int]]:
        return decode_action_id(self.action_id)


class TensorTransitionDataset(Sequence[TensorTransitionRecord]):
    """Lightweight loader for offline RL tensor transition archives."""

    def __init__(self, prefix: Path | str):
        base = Path(prefix)
        if base.suffix:
            base = base.with_suffix("")
        self._base = base
        npz_path = base.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Tensor archive not found: {npz_path}")

        meta_path = base.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Tensor metadata not found: {meta_path}")

        self._archive = np.load(npz_path, allow_pickle=False)

        self._states = self._archive["states"]
        self._next_states = self._archive["next_states"]
        self._rewards = self._archive["rewards"]
        self._dones = self._archive["dones"]
        if "actions" in self._archive.keys():
            self._actions = self._archive["actions"].astype(np.int64)
        else:
            self._actions = np.full(len(self._states), -1, dtype=np.int64)

        self._metadata: List[Dict[str, object]] = json.loads(meta_path.read_text())
        self._validate_lengths()

        self._num_players = _infer_num_players(self._states[0]) if len(self) else 0
        self._num_ranks = self._states.shape[1] if len(self) else 13

    def _validate_lengths(self) -> None:
        expected = len(self._states)
        for name, array in (
            ("next_states", self._next_states),
            ("rewards", self._rewards),
            ("dones", self._dones),
            ("actions", self._actions),
        ):
            if len(array) != expected:
                raise ValueError(f"Archive array '{name}' has mismatched length.")

        if len(self._metadata) != expected:
            raise ValueError("Metadata length does not match tensor archive.")

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, index: int) -> TensorTransitionRecord:
        meta = dict(self._metadata[index])
        action_id = int(self._actions[index])
        return TensorTransitionRecord(
            index=int(meta.get("index", index)),
            state=self._states[index],
            action_id=action_id,
            reward=float(self._rewards[index]),
            next_state=self._next_states[index],
            done=bool(self._dones[index]),
            metadata=meta,
        )

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def num_ranks(self) -> int:
        return self._num_ranks

    def iter_records(self) -> Iterator[TensorTransitionRecord]:
        for idx in range(len(self)):
            yield self[idx]

    def as_qtransformer_arrays(self) -> Dict[str, np.ndarray]:
        """Return numpy arrays aligned with QTransformer expectations."""

        if not len(self):
            raise ValueError("Tensor archive is empty; nothing to convert.")

        player_cards: List[np.ndarray] = []
        holding_cards: List[int] = []
        discard_cards: List[int] = []
        next_player_cards: List[np.ndarray] = []
        next_holding_cards: List[int] = []
        next_discard_cards: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []

        for record in self.iter_records():
            player_id = int(record.metadata.get("player_id", -1))
            if player_id < 0:
                raise ValueError(
                    "player_id missing from metadata; cannot build QTransformer inputs."
                )

            cards, holding, discard_top = tensor_to_player_tokens(
                record.state,
                player_id=player_id,
                num_players=self._num_players,
            )
            nxt_cards, nxt_holding, nxt_discard = tensor_to_player_tokens(
                record.next_state,
                player_id=player_id,
                num_players=self._num_players,
            )

            player_cards.append(cards)
            holding_cards.append(holding)
            discard_cards.append(discard_top)
            next_player_cards.append(nxt_cards)
            next_holding_cards.append(nxt_holding)
            next_discard_cards.append(nxt_discard)
            actions.append(int(record.action_id))
            rewards.append(float(record.reward))
            dones.append(bool(record.done))

        return {
            "player_cards": np.stack(player_cards, axis=0).astype(np.int64),
            "holding_cards": np.asarray(holding_cards, dtype=np.int64),
            "discard_top": np.asarray(discard_cards, dtype=np.int64)[:, None],
            "next_player_cards": np.stack(next_player_cards, axis=0).astype(np.int64),
            "next_holding_cards": np.asarray(next_holding_cards, dtype=np.int64),
            "next_discard_top": np.asarray(next_discard_cards, dtype=np.int64)[:, None],
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.bool_),
        }

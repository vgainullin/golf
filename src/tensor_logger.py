"""Utilities for collecting tensor-based Golf transitions."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import numpy as np


@dataclass
class TransitionMetadata:
    """Lightweight description of a single transition."""

    index: int
    game: int
    hole: int
    round_num: int
    player_id: int
    action_num: int
    action: Optional[int]
    position: Optional[int]
    reward: float
    done: bool


class TensorTransitionLogger:
    """Accumulates tensor transitions and writes them to disk."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._states: List[np.ndarray] = []
        self._next_states: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._metadata: List[TransitionMetadata] = []

        # Metric tracking placeholders; populated as logging occurs.
        self._unique_hashes: set[bytes] = set()
        self._rank_counts: Optional[np.ndarray] = None
        self._position_counts: Optional[np.ndarray] = None
        self._suit_counts: Optional[np.ndarray] = None
        self._transition_hamming: List[int] = []
        self._reward_sum: float = 0.0
        self._reward_sq_sum: float = 0.0
        self._reward_count: int = 0

    def log(
        self,
        *,
        state: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        metadata: Dict[str, Any],
    ) -> None:
        """Record a single transition."""
        if state.shape != next_state.shape:
            raise ValueError("State and next_state tensors must share the same shape")

        index = len(self._states)
        self._states.append(state.astype(np.int8, copy=False))
        self._next_states.append(next_state.astype(np.int8, copy=False))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._metadata.append(
            TransitionMetadata(
                index=index,
                game=int(metadata.get("game", -1)),
                hole=int(metadata.get("hole", -1)),
                round_num=int(metadata.get("round", -1)),
                player_id=int(metadata.get("player_id", -1)),
                action_num=int(metadata.get("action_num", -1)),
                action=(
                    None
                    if metadata.get("action") is None
                    else int(metadata["action"])
                ),
                position=(
                    None
                    if metadata.get("position") is None
                    else int(metadata["position"])
                ),
                reward=float(reward),
                done=bool(done),
            )
        )

    def __len__(self) -> int:
        return len(self._states)

    def save(self, *, prefix: str = "tensor_transitions") -> None:
        """Persist all logged transitions to disk."""
        if not self._states:
            return

        base_path = self.output_dir / prefix
        np.savez_compressed(
            base_path.with_suffix(".npz"),
            states=np.stack(self._states, axis=0),
            next_states=np.stack(self._next_states, axis=0),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.bool_),
        )

        metadata_path = base_path.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump([asdict(item) for item in self._metadata], fh, indent=2)

    def clear(self) -> None:
        """Reset the logger state (useful for tests)."""
        self._states.clear()
        self._next_states.clear()
        self._rewards.clear()
        self._dones.clear()
        self._metadata.clear()

    def extend(self, other: "TensorTransitionLogger") -> None:
        """Merge records from another logger into this one."""
        offset = len(self._states)
        for record in other._metadata:
            merged_record = TransitionMetadata(**asdict(record))
            merged_record.index += offset
            self._metadata.append(merged_record)

        self._states.extend(other._states)
        self._next_states.extend(other._next_states)
        self._rewards.extend(other._rewards)
        self._dones.extend(other._dones)

    @property
    def metadata(self) -> Iterable[TransitionMetadata]:
        return tuple(self._metadata)

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return a snapshot of accumulated metrics."""

        reward_mean = (
            self._reward_sum / self._reward_count if self._reward_count else 0.0
        )
        reward_var = (
            (self._reward_sq_sum / self._reward_count) - reward_mean ** 2
            if self._reward_count
            else 0.0
        )

        def _entropy(counts: Optional[np.ndarray]) -> float:
            if counts is None:
                return 0.0
            total = counts.sum()
            if not total:
                return 0.0
            probs = counts / total
            non_zero = probs[probs > 0]
            return float(-(non_zero * np.log2(non_zero)).sum())

        return {
            "unique_states": len(self._unique_hashes),
            "total_states": len(self._states),
            "entropy_rank": _entropy(self._rank_counts),
            "entropy_position": _entropy(self._position_counts),
            "entropy_suit": _entropy(self._suit_counts),
            "avg_transition_hamming": (
                float(np.mean(self._transition_hamming))
                if self._transition_hamming
                else 0.0
            ),
            "reward_mean": reward_mean,
            "reward_variance": reward_var,
        }

"""Utilities for loading and using the offline-trained DQN agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .dqn_offline import (
    GolfDQN,
    TrainingConfig,
    NUM_ACTIONS,
    STATE_SEQUENCE_LENGTH,
    STAGE0_LEGAL,
    STAGE1_LEGAL,
)


def _resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class OfflineDQAgent:
    """Wrapper around the offline-trained DQN checkpoint for inference."""

    def __init__(self, checkpoint_path: Path | str, *, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Try to load with weights_only=True first for security
        try:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            # Fall back to weights_only=False if checkpoint contains Python objects
            # SECURITY WARNING: This allows arbitrary code execution from untrusted checkpoints
            import warnings
            warnings.warn(
                f"Loading checkpoint {self.checkpoint_path} with weights_only=False. "
                "Only load checkpoints from trusted sources.",
                RuntimeWarning,
                stacklevel=2,
            )
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Validate checkpoint structure
        required_keys = {"model_state_dict", "config"}
        if not required_keys.issubset(state.keys()):
            raise ValueError(
                f"Checkpoint is missing required keys. Expected {required_keys}, "
                f"found {set(state.keys())}"
            )

        config_dict = state.get("config")
        if config_dict is None:
            raise ValueError("Checkpoint is missing TrainingConfig data.")

        # Validate config structure
        if not isinstance(config_dict, dict):
            raise ValueError(f"Config must be a dict, got {type(config_dict)}")

        config_kwargs = dict(config_dict)
        config_kwargs["archive_prefix"] = Path(config_kwargs["archive_prefix"])
        config_kwargs["output_dir"] = Path(config_kwargs["output_dir"])
        self.config = TrainingConfig(**config_kwargs)

        self.device = _resolve_device(device)
        self.model = GolfDQN(self.config.embedding_dim, self.config.hidden_dim)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._action_masks = {
            0: STAGE0_LEGAL.to(self.device),
            1: STAGE1_LEGAL.to(self.device),
        }

    def _forward(self, state_tokens: np.ndarray, stage_index: int) -> Tensor:
        if state_tokens.shape[0] != STATE_SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected {STATE_SEQUENCE_LENGTH} tokens (6 cards + holding + discard), got {state_tokens.shape[0]}."
            )
        state_tensor = torch.as_tensor(state_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        stage_tensor = torch.as_tensor([stage_index], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model(state_tensor, stage_tensor)

    def select_action(self, state_tokens: np.ndarray, action_stage: int) -> int:
        """Return the action id favored by the policy for the given stage (0 or 1)."""
        if action_stage not in self._action_masks:
            raise ValueError(f"Unsupported action_stage {action_stage}; expected 0 or 1.")
        q_values = self._forward(state_tokens, action_stage)
        mask = self._action_masks[action_stage]
        masked = q_values.masked_fill(~mask.unsqueeze(0), float("-inf"))
        action_idx = int(torch.argmax(masked, dim=1).item())
        return action_idx

    def q_values(self, state_tokens: np.ndarray, action_stage: int) -> Tensor:
        """Return masked Q-values for the specified stage."""
        if action_stage not in self._action_masks:
            raise ValueError(f"Unsupported action_stage {action_stage}; expected 0 or 1.")
        q_values = self._forward(state_tokens, action_stage)
        mask = self._action_masks[action_stage]
        return q_values.masked_fill(~mask.unsqueeze(0), float("-inf")).squeeze(0)

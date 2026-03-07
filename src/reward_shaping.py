"""Reward shaping for DQN training loops.

Corrects the systematic bias in stage-1 rewards where compute_score treats
unrevealed cards as 0, creating a +5.2 point bias toward revealed positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .vectorized_golf import compute_final_score


class RewardShaper(ABC):
    """Base class for reward correction in DQN training loops."""

    @abstractmethod
    def shape(self, reward: torch.Tensor, **context) -> torch.Tensor:
        """Return corrected reward given environment context."""


class HindsightRewardShaper(RewardShaper):
    """Corrects stage-1 rewards using true card values (hindsight).

    The agent plays with partial information (can't see unrevealed cards),
    but learns from full information. After each stage-1 action, the true
    card values are already in state.player_cards -- we compute reward
    using compute_final_score (all cards revealed) instead of compute_score
    (only revealed cards count).
    """

    def shape(
        self,
        reward: torch.Tensor,
        *,
        cards_before: torch.Tensor,
        cards_after: torch.Tensor,
        active: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        true_before = compute_final_score(cards_before, device)
        true_after = compute_final_score(cards_after, device)
        shaped = true_before - true_after
        return torch.where(active, shaped, torch.zeros_like(shaped))

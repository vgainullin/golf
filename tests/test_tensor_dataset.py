import numpy as np
import pytest

from src.tensor_dataset import (
    TensorTransitionDataset,
    tensor_to_player_tokens,
)
from src.tensor_logger import TensorTransitionLogger, decode_action_id


def test_tensor_transition_dataset_roundtrip(tmp_path):
    logger = TensorTransitionLogger(tmp_path)

    state = np.zeros((13, 10, 4), dtype=np.int8)
    next_state = np.zeros_like(state)

    # Player 0 cards (positions 0-5)
    state[0, 0, 0] = 1  # 2 of spades
    state[1, 1, 1] = 1  # 3 of diamonds
    state[2, 2, 2] = 1  # 4 of clubs
    state[3, 3, 3] = 1  # 5 of hearts
    state[4, 4, 0] = 1
    state[5, 5, 1] = 1
    state[6, 6, 2] = 1  # holding slot (position 6)

    # Deck column (position 7) not used

    # Discard pile (position 8) - leave empty

    # Face card (position 9)
    state[7, 9, 3] = 1

    # Next state modifications
    next_state[:] = state
    next_state[7, 9, 3] = 0
    next_state[8, 9, 0] = 1

    logger.log(
        state=state,
        next_state=next_state,
        reward=2.5,
        done=False,
        metadata={
            "game": 1,
            "hole": 2,
            "round": 3,
            "player_id": 0,
            "action_num": 1,
            "action": 0,
            "position": 2,
        },
    )

    logger.save(prefix="dataset_case")

    ds = TensorTransitionDataset(tmp_path / "dataset_case")
    assert len(ds) == 1

    record = ds[0]
    assert record.index == 0
    assert record.action_id == 4  # action_num=1, action=0, position=2
    assert decode_action_id(record.action_id) == (1, 0, 2)

    q_arrays = ds.as_qtransformer_arrays()
    assert q_arrays["player_cards"].shape == (1, 6)
    assert q_arrays["discard_top"].shape == (1, 1)
    assert q_arrays["next_discard_top"].shape == (1, 1)
    assert q_arrays["actions"].tolist() == [4]
    assert q_arrays["rewards"][0] == pytest.approx(2.5)

    # Validate specific card indices
    # 2 of spades -> suit 0, rank 0 -> index 0
    assert q_arrays["player_cards"][0, 0] == 0
    # Holding slot should reflect slot 6 (clubs rank 6 -> 6 of clubs -> suit 2)
    assert q_arrays["holding_cards"][0] == 2 * 13 + 6


def test_tensor_to_player_tokens_handles_unknown(tmp_path):
    state = np.zeros((13, 10, 4), dtype=np.int8)
    # Only face card known
    state[0, 9, 0] = 1
    cards, holding, discard_top = tensor_to_player_tokens(state, player_id=0, num_players=1)
    assert cards.tolist() == [52, 52, 52, 52, 52, 52]
    assert holding == 52
    assert discard_top == 0

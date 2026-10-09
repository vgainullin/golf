"""Regression tests for the bounded posterior-rollout lookahead player."""

from dataclasses import fields

import torch

from src.bayes_optimal import (
    BayesBeliefTracker,
    bounded_lookahead_stage0,
    bounded_lookahead_stage1,
    lookahead_stage0,
    lookahead_stage1,
)
from src.vectorized_golf import VectorizedGolfState, reset_games


DEVICE = torch.device("cpu")


def _clone_state(state: VectorizedGolfState) -> VectorizedGolfState:
    values = {}
    for field in fields(state):
        value = getattr(state, field.name)
        values[field.name] = value.clone() if torch.is_tensor(value) else value
    return VectorizedGolfState(**values)


def _assert_state_equal(
    actual: VectorizedGolfState, expected: VectorizedGolfState
) -> None:
    for field in fields(actual):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if torch.is_tensor(actual_value):
            assert torch.equal(actual_value, expected_value), field.name
        else:
            assert actual_value == expected_value, field.name


def _stage1_state(N: int = 32) -> VectorizedGolfState:
    state = reset_games(N=N, device=DEVICE)
    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top
    return state


def _forced_multi_flip_state(N: int = 32) -> VectorizedGolfState:
    """A layout where a bad held card should be discarded with 3 flip choices."""
    state = _stage1_state(N)
    # J, Q, and K are face up in the top row. The differing partner ranks make
    # the flip candidates observably different while a held Ten still worsens
    # every column, forcing the bounded-rollout branch.
    state.player_cards[:, 0, 0] = 9
    state.player_cards[:, 0, 1] = 10
    state.player_cards[:, 0, 2] = 11
    state.player_revealed[:, 0, :3] = True
    state.player_revealed[:, 0, 3:] = False
    state.player_holding[:, 0] = 8
    return state


def _forced_finish_state(N: int = 32) -> VectorizedGolfState:
    """A layout where L would flip its final card and end the hole."""
    state = _stage1_state(N)
    # This public layout is a reduced fixture from a seeded live game. L flips
    # slot 5; the bounded continuation instead places into a revealed slot.
    state.player_cards[:, 0, :5] = torch.tensor(
        [1, 39, 28, 2, 32], dtype=state.player_cards.dtype
    )
    state.player_revealed[:, 0, :5] = True
    state.player_revealed[:, 0, 5] = False
    state.discard_top.fill_(9)
    state.player_holding[:, 0] = 9
    state.last_turn.fill_(False)
    return state


def _forced_place_finish_state(N: int = 32) -> VectorizedGolfState:
    """A live-game fixture where L places into its final hidden slot."""
    state = _stage1_state(N)
    state.player_cards[:, 0, :5] = torch.tensor(
        [24, 43, 37, 16, 17], dtype=state.player_cards.dtype
    )
    state.player_revealed[:, 0, :5] = True
    state.player_revealed[:, 0, 5] = False
    state.discard_top.fill_(12)
    state.player_holding[:, 0] = 12
    state.last_turn.fill_(False)
    return state


def _tracker(state: VectorizedGolfState) -> BayesBeliefTracker:
    tracker = BayesBeliefTracker(state.player_cards.shape[0], DEVICE)
    tracker.observe(state, my_player_id=0)
    return tracker


def test_bounded_stage0_is_exactly_lookahead_stage0():
    torch.manual_seed(101)
    state = reset_games(N=64, device=DEVICE)
    state.player_revealed[:, 0, 0] = True
    tracker = _tracker(state)

    expected = lookahead_stage0(state, 0, tracker)
    actual = bounded_lookahead_stage0(state, 0, tracker)

    assert torch.equal(actual, expected)
    assert ((actual == 0) | (actual == 1)).all()


def test_disabled_continuation_is_exactly_lookahead_stage1():
    torch.manual_seed(102)
    state = _stage1_state(N=64)
    state.player_revealed[:, 0, :2] = True
    tracker = _tracker(state)

    expected = lookahead_stage1(state, 0, tracker)
    actual = bounded_lookahead_stage1(state, 0, tracker, enabled=False)

    assert torch.equal(actual, expected)


def test_bounded_stage1_actions_are_legal():
    torch.manual_seed(103)
    state = _forced_finish_state(N=64)
    tracker = _tracker(state)
    base_action = lookahead_stage1(state, 0, tracker)
    action = bounded_lookahead_stage1(state, 0, tracker)

    valid = ((action >= 2) & (action <= 7)) | ((action >= 9) & (action <= 14))
    assert action.shape == (64,)
    assert valid.all()
    assert (base_action == 14).all()
    assert ((action >= 2) & (action <= 6)).all()


def test_bounded_stage1_is_deterministic():
    torch.manual_seed(104)
    state = _forced_finish_state(N=48)
    tracker = _tracker(state)

    first = bounded_lookahead_stage1(state, 0, tracker)
    second = bounded_lookahead_stage1(state, 0, tracker)

    assert torch.equal(first, second)


def test_bounded_stage1_can_delay_a_finishing_placement():
    state = _forced_place_finish_state(N=16)
    tracker = _tracker(state)

    base_action = lookahead_stage1(state, 0, tracker)
    action = bounded_lookahead_stage1(state, 0, tracker)

    assert (base_action == 7).all()
    assert ((action >= 2) & (action <= 6)).all()


def test_multiple_hidden_cards_fall_back_to_lookahead_stage1():
    state = _forced_multi_flip_state(N=16)
    tracker = _tracker(state)

    expected = lookahead_stage1(state, 0, tracker)
    actual = bounded_lookahead_stage1(state, 0, tracker)

    assert torch.equal(actual, expected)


def test_bounded_stage1_does_not_advance_global_rng():
    torch.manual_seed(105)
    state = _forced_finish_state(N=24)
    tracker = _tracker(state)
    global_rng_before = torch.random.get_rng_state().clone()

    bounded_lookahead_stage1(state, 0, tracker)

    assert torch.equal(torch.random.get_rng_state(), global_rng_before)


def test_bounded_stage1_does_not_peek_at_true_hidden_cards():
    torch.manual_seed(106)
    original = _forced_finish_state(N=40)
    scrambled = _clone_state(original)
    # Only latent identities change. Public state and the posterior are fixed.
    scrambled.player_cards[:, 0, 5] = original.player_cards[:, 0, 5].roll(
        shifts=1, dims=0
    )
    assert not torch.equal(
        original.player_cards[:, 0, 5], scrambled.player_cards[:, 0, 5]
    )
    tracker = _tracker(original)

    original_action = bounded_lookahead_stage1(
        original,
        0,
        tracker,
    )
    scrambled_action = bounded_lookahead_stage1(
        scrambled,
        0,
        tracker,
    )

    assert torch.equal(original_action, scrambled_action)


def test_bounded_stage1_does_not_mutate_state_or_belief():
    torch.manual_seed(107)
    state = _forced_finish_state(N=24)
    tracker = _tracker(state)
    # Materialize the lazy rank cache before taking the mutation snapshot.
    tracker.multiset_by_rank()
    state_before = _clone_state(state)
    belief_before = tracker.unobserved.clone()

    bounded_lookahead_stage1(state, 0, tracker)

    _assert_state_equal(state, state_before)
    assert torch.equal(tracker.unobserved, belief_before)


def test_active_last_round_falls_back_to_lookahead_stage1():
    torch.manual_seed(108)
    state = _stage1_state(N=20)
    state.player_revealed[:, 0, :5] = True
    state.player_revealed[:, 0, 5] = False
    state.last_turn.fill_(True)
    tracker = _tracker(state)

    expected = lookahead_stage1(state, 0, tracker)
    actual = bounded_lookahead_stage1(
        state,
        0,
        tracker,
    )

    assert torch.equal(actual, expected)


def test_post_reshuffle_state_falls_back_to_lookahead_stage1():
    state = _forced_finish_state(N=20)
    state.deck_size.fill_(12)
    tracker = _tracker(state)

    expected = lookahead_stage1(state, 0, tracker)
    actual = bounded_lookahead_stage1(state, 0, tracker)

    assert torch.equal(actual, expected)


def test_exhausted_deck_falls_back_to_lookahead_stage1():
    state = _forced_finish_state(N=20)
    state.deck_ptr.copy_(state.deck_size)
    tracker = _tracker(state)

    expected = lookahead_stage1(state, 0, tracker)
    actual = bounded_lookahead_stage1(state, 0, tracker)

    assert torch.equal(actual, expected)


def test_imminent_reshuffle_falls_back_to_lookahead_stage1():
    state = _forced_finish_state(N=20)
    state.deck_ptr.copy_(state.deck_size - (state.n_players - 1))
    tracker = _tracker(state)

    expected = lookahead_stage1(state, 0, tracker)
    actual = bounded_lookahead_stage1(state, 0, tracker)

    assert torch.equal(actual, expected)

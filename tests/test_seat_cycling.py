"""Focused integration tests for the seat-cycling agent harness."""

import torch

from scripts.seat_cycling import SeatHandler, run_seating, unique_permutations
from src.bayes_optimal import BayesBeliefTracker
from src.vectorized_golf import reset_games


DEVICE = torch.device("cpu")


def test_l2_is_a_supported_seat_label():
    assert "L2" in SeatHandler.LABELS


def test_l2_l_r_r_permutations_are_balanced():
    permutations = unique_permutations(["L2", "L", "R", "R"])

    assert len(permutations) == 12
    for seat in range(4):
        assert sum(permutation[seat] == "L2" for permutation in permutations) == 3
        assert sum(permutation[seat] == "L" for permutation in permutations) == 3


def test_l2_handlers_have_independent_trackers():
    first = SeatHandler("L2", seat_idx=0, N=2, device=DEVICE)
    second = SeatHandler("L2", seat_idx=0, N=2, device=DEVICE)

    assert isinstance(first.tracker, BayesBeliefTracker)
    assert first.tracker is not second.tracker


def test_disabled_l2_continuation_matches_l_actions(monkeypatch):
    monkeypatch.setattr(SeatHandler, "l2_enabled", False)
    torch.manual_seed(7)
    state = reset_games(N=16, device=DEVICE)
    lookahead = SeatHandler("L", seat_idx=0, N=16, device=DEVICE)
    bounded = SeatHandler("L2", seat_idx=0, N=16, device=DEVICE)
    lookahead.reset_for_hole(state)
    bounded.reset_for_hole(state)

    assert torch.equal(lookahead.stage0(state), bounded.stage0(state))

    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top
    assert torch.equal(lookahead.stage1(state), bounded.stage1(state))


def test_run_seating_with_l2_one_hole_smoke(monkeypatch):
    monkeypatch.setattr(SeatHandler, "l2_enabled", True)
    torch.manual_seed(11)

    scores, win_rates = run_seating(
        ("L2", "L", "R", "R"),
        num_games=2,
        holes=1,
        device=DEVICE,
    )

    assert len(scores) == 4
    assert len(win_rates) == 4
    assert torch.isfinite(torch.tensor(scores)).all()
    assert torch.isfinite(torch.tensor(win_rates)).all()
    assert abs(sum(win_rates) - 1.0) < 1e-6

"""Tests for the belief-augmented Bayes player."""
import torch

from src.bayes_optimal import (
    BayesBeliefTracker,
    bayes_stage0,
    bayes_stage1,
    expected_score,
    run_bayes_eval,
)
from src.vectorized_golf import (
    NUM_CARDS,
    NUM_RANKS,
    RANK_SCORES,
    compute_final_score,
    compute_score,
    reset_games,
)


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Belief tracker
# ---------------------------------------------------------------------------


def test_tracker_starts_with_all_cards_unobserved():
    t = BayesBeliefTracker(N=3, device=DEVICE)
    assert t.unobserved.shape == (3, 52)
    assert t.unobserved.all()
    assert (t.total() == 52).all()


def test_tracker_observes_revealed_cards_and_discard_top():
    state = reset_games(N=2, device=DEVICE)
    # Reveal slot 0 for player 0 in both games
    state.player_revealed[:, 0, 0] = True
    revealed_cards = state.player_cards[:, 0, 0].long()  # (2,)
    discard_cards = state.discard_top.long()  # (2,)

    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state)

    for n in range(2):
        assert not t.unobserved[n, revealed_cards[n]]
        assert not t.unobserved[n, discard_cards[n]]

    # Should have observed exactly: 1 revealed + 1 discard top = 2 distinct
    # cards (assuming they aren't the same; reset gives distinct cards).
    assert (t.total() == 50).all()


def test_tracker_is_idempotent():
    state = reset_games(N=2, device=DEVICE)
    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state)
    snapshot = t.unobserved.clone()
    t.observe(state)
    t.observe(state)
    assert torch.equal(t.unobserved, snapshot)


def test_tracker_reset_restores_full_belief():
    state = reset_games(N=2, device=DEVICE)
    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state)
    assert (t.total() < 52).any()
    t.reset()
    assert (t.total() == 52).all()


def test_multiset_sums_to_total():
    state = reset_games(N=4, device=DEVICE)
    state.player_revealed[:, :, :3] = True  # reveal 3 slots for everyone
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state)
    multiset = t.multiset_by_rank()
    assert multiset.shape == (4, 13)
    assert torch.equal(multiset.sum(dim=1), t.total())


def test_expected_unknown_score_full_deck_matches_global_average():
    t = BayesBeliefTracker(N=1, device=DEVICE)
    e = t.expected_unknown_score()
    # Full deck: 4 of each rank, equal weight => mean of RANK_SCORES.
    expected = RANK_SCORES.mean().item()
    assert abs(e.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# expected_score math
# ---------------------------------------------------------------------------


def _full_deck_belief(N: int) -> tuple[torch.Tensor, torch.Tensor]:
    multiset = torch.full((N, NUM_RANKS), 4, dtype=torch.int64)
    total = torch.full((N,), NUM_CARDS, dtype=torch.int64)
    return multiset, total


def test_expected_score_fully_revealed_matches_compute_score():
    state = reset_games(N=8, device=DEVICE)
    state.player_revealed[:] = True  # everyone fully revealed
    cards = state.player_cards[:, 0, :]
    revealed = state.player_revealed[:, 0, :]

    multiset, total = _full_deck_belief(8)
    e = expected_score(cards, revealed, multiset, total, DEVICE)
    cs = compute_score(cards, revealed, DEVICE)
    assert torch.allclose(e, cs)


def test_expected_score_fully_hidden_layout_is_three_columns():
    # 6 face-down cards, full deck belief.
    # Each column both-hidden expectation should be:
    #   2 * E[?] - 2 * sum_r P(both=r)*score(r)
    # E[?] = mean(RANK_SCORES) = 4.85... (but use exact formula)
    cards = torch.zeros(1, 6, dtype=torch.int16)
    revealed = torch.zeros(1, 6, dtype=torch.bool)
    multiset, total = _full_deck_belief(1)

    e = expected_score(cards, revealed, multiset, total, DEVICE)

    # Compute expected value of one column manually.
    rs = RANK_SCORES  # (13,)
    e_unknown = rs.mean()  # 4 of each rank, uniform
    # P(both same rank, w/o replacement) per rank: 4*3/(52*51) = 12/2652
    p_per_rank = (4 * 3) / (52 * 51)
    correction = sum(p_per_rank * s.item() for s in rs)
    e_col = 2 * e_unknown.item() - 2 * correction
    expected_total = 3 * e_col

    assert abs(e.item() - expected_total) < 1e-4


def test_expected_score_one_revealed_one_hidden_column():
    # Column 0: slot 0 = K (rank 11, score 0), slot 3 hidden.
    # P(? = K) under full deck = 4/52
    # E[col] = (1 - 4/52) * (0 + E[score(?) | ? != K])
    # E[score(?) | ? != K] = (sum_r 4*score(r) - 4*0) / 48 = 4 * sum_{r != K} score(r) / 48
    cards = torch.zeros(1, 6, dtype=torch.int16)
    cards[0, 0] = 11  # rank K (suit 0)
    revealed = torch.zeros(1, 6, dtype=torch.bool)
    revealed[0, 0] = True
    # Other columns (1, 2) both hidden, expected per-col same as above.
    multiset, total = _full_deck_belief(1)
    e = expected_score(cards, revealed, multiset, total, DEVICE)

    rs = RANK_SCORES
    p_match = 4 / 52
    sum_n_score = 4 * rs.sum().item()
    e_other = (sum_n_score - 4 * 0) / (52 - 4)
    e_col0 = (1 - p_match) * (0 + e_other)

    e_unknown = rs.mean().item()
    p_per_rank = (4 * 3) / (52 * 51)
    correction = sum(p_per_rank * s.item() for s in rs)
    e_col_hidden = 2 * e_unknown - 2 * correction

    expected = e_col0 + 2 * e_col_hidden
    assert abs(e.item() - expected) < 1e-4


def test_expected_score_column_match_zeroes_column():
    # Both slots in col 0 = K (rank 11, score 0); should be 0 (column-match).
    cards = torch.zeros(1, 6, dtype=torch.int16)
    cards[0, 0] = 11  # K of suit 0
    cards[0, 3] = 11 + NUM_RANKS  # K of suit 1
    revealed = torch.zeros(1, 6, dtype=torch.bool)
    revealed[0, 0] = True
    revealed[0, 3] = True
    multiset, total = _full_deck_belief(1)

    e = expected_score(cards, revealed, multiset, total, DEVICE)

    rs = RANK_SCORES
    e_unknown = rs.mean().item()
    p_per_rank = (4 * 3) / (52 * 51)
    correction = sum(p_per_rank * s.item() for s in rs)
    e_col_hidden = 2 * e_unknown - 2 * correction

    expected = 0 + 2 * e_col_hidden  # col 0 = 0, cols 1/2 hidden
    assert abs(e.item() - expected) < 1e-4


# ---------------------------------------------------------------------------
# Stage functions sanity
# ---------------------------------------------------------------------------


def test_bayes_stage0_returns_valid_action():
    state = reset_games(N=4, device=DEVICE)
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state)
    a = bayes_stage0(state, 0, t)
    assert a.shape == (4,)
    assert ((a == 0) | (a == 1)).all()


def test_bayes_stage1_returns_valid_action():
    state = reset_games(N=4, device=DEVICE)
    # Put a card in the holding slot for player 0
    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state)
    a = bayes_stage1(state, 0, t)
    assert a.shape == (4,)
    # action is 2-7 (place) or 9-14 (discard+flip)
    valid = ((a >= 2) & (a <= 7)) | ((a >= 9) & (a <= 14))
    assert valid.all()


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_bayes_player_beats_random_opponents():
    torch.manual_seed(0)
    score = run_bayes_eval(
        opponent_specs=["R", "R", "R"],
        num_games=200,
        holes=3,
        device=DEVICE,
    )
    # Random baseline is ~31, improved heuristic ~10. Bayes player should
    # easily beat random; expect well under 15.
    assert score < 15.0, f"bayes seat 0 avg {score:.2f} should beat random easily"

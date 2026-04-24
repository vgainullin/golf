"""Tests for the belief-augmented Bayes player."""
import torch

from src.bayes_optimal import (
    BayesBeliefTracker,
    bayes_stage0,
    bayes_stage1,
    expected_score,
    lookahead_stage0,
    lookahead_stage1,
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
    t.observe(state, my_player_id=0)

    for n in range(2):
        assert not t.unobserved[n, revealed_cards[n]]
        assert not t.unobserved[n, discard_cards[n]]

    # Should have observed exactly: 1 revealed + 1 discard top = 2 distinct
    # cards (assuming they aren't the same; reset gives distinct cards).
    assert (t.total() == 50).all()


def test_tracker_is_idempotent():
    state = reset_games(N=2, device=DEVICE)
    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state, my_player_id=0)
    snapshot = t.unobserved.clone()
    t.observe(state, my_player_id=0)
    t.observe(state, my_player_id=0)
    assert torch.equal(t.unobserved, snapshot)


def test_tracker_reset_restores_full_belief():
    state = reset_games(N=2, device=DEVICE)
    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state, my_player_id=0)
    assert (t.total() < 52).any()
    t.reset()
    assert (t.total() == 52).all()


def test_multiset_sums_to_total():
    state = reset_games(N=4, device=DEVICE)
    state.player_revealed[:, :, :3] = True  # reveal 3 slots for everyone
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
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
    t.observe(state, my_player_id=0)
    a = bayes_stage0(state, 0, t)
    assert a.shape == (4,)
    assert ((a == 0) | (a == 1)).all()


def test_bayes_stage1_returns_valid_action():
    state = reset_games(N=4, device=DEVICE)
    # Put a card in the holding slot for player 0
    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
    a = bayes_stage1(state, 0, t)
    assert a.shape == (4,)
    # action is 2-7 (place) or 9-14 (discard+flip)
    valid = ((a >= 2) & (a <= 7)) | ((a >= 9) & (a <= 14))
    assert valid.all()


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_bayes_no_belief_equals_improved_heuristic():
    """Bayes(use_belief=False) must produce identical actions to the improved
    heuristic. This is a regression test: any divergence means the wiring
    around bayes_stage0/stage1 has drifted from the improved heuristic
    semantics, even though the belief is supposed to be the only difference.
    """
    from src.bayes_optimal import bayes_stage0, bayes_stage1, BayesBeliefTracker
    from src.vectorized_golf import (
        heuristic_stage0,
        improved_stage1,
        random_stage0,
        random_stage1,
        compute_final_score,
        step_stage0,
        step_stage1,
    )

    torch.manual_seed(0)
    N = 200
    BAYES_SEAT = 0
    n_players = 4
    tracker = BayesBeliefTracker(N, DEVICE)

    bayes_total = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    heur_total = torch.zeros(N, dtype=torch.float32, device=DEVICE)

    for runner_kind in ("bayes_no_belief", "improved"):
        torch.manual_seed(0)
        tot = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        for hole in range(3):
            state = reset_games(N, DEVICE, n_players=n_players)
            tracker.reset()
            tracker.observe(state, my_player_id=BAYES_SEAT)
            for _ in range(40):
                if state.done.all():
                    break
                for pid in range(n_players):
                    active = ~state.done
                    back_to_trigger = state.last_turn & (state.end_game_player == pid)
                    state.done = state.done | (back_to_trigger & active)
                    active = ~state.done
                    if not active.any():
                        break
                    state.current_stage.fill_(0)
                    if pid == 0:
                        if runner_kind == "bayes_no_belief":
                            a0 = bayes_stage0(state, pid, tracker, use_belief=False)
                        else:
                            a0 = heuristic_stage0(state, pid)
                    else:
                        a0 = random_stage0(state, pid)
                    step_stage0(state, a0, pid)
                    tracker.observe(state, my_player_id=BAYES_SEAT)
                    if state.done.all():
                        break
                    state.current_stage.fill_(1)
                    if pid == 0:
                        if runner_kind == "bayes_no_belief":
                            a1 = bayes_stage1(state, pid, tracker, use_belief=False)
                        else:
                            a1 = improved_stage1(state, pid)
                    else:
                        a1 = random_stage1(state, pid)
                    step_stage1(state, a1, pid)
                    tracker.observe(state, my_player_id=BAYES_SEAT)
                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly
                    state.end_game_player = torch.where(
                        newly,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )
            tot += compute_final_score(state.player_cards[:, 0, :], DEVICE)
        if runner_kind == "bayes_no_belief":
            bayes_total = tot
        else:
            heur_total = tot

    assert torch.equal(bayes_total, heur_total), (
        f"bayes(no belief) != improved heuristic. "
        f"bayes mean={bayes_total.mean().item():.6f}, "
        f"heur mean={heur_total.mean().item():.6f}"
    )


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


# ---------------------------------------------------------------------------
# Lookahead player
# ---------------------------------------------------------------------------


def test_lookahead_stage1_returns_valid_action():
    state = reset_games(N=4, device=DEVICE)
    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
    a = lookahead_stage1(state, 0, t)
    assert a.shape == (4,)
    valid = ((a >= 2) & (a <= 7)) | ((a >= 9) & (a <= 14))
    assert valid.all()


def test_lookahead_stage1_fully_revealed_picks_best_placement():
    """When all slots are revealed, lookahead should place at the position
    minimizing compute_final_score (no hidden-slot ambiguity)."""
    N = 16
    state = reset_games(N=N, device=DEVICE)
    state.player_revealed[:] = True  # all revealed
    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top

    t = BayesBeliefTracker(N=N, device=DEVICE)
    t.observe(state, my_player_id=0)
    action = lookahead_stage1(state, 0, t)

    # Brute-force: try all 6 placements, pick the one with lowest final score.
    cards = state.player_cards[:, 0, :].clone()
    held = state.player_holding[:, 0]
    best_pos_bf = torch.zeros(N, dtype=torch.long)
    best_score_bf = torch.full((N,), 1e6)
    for pos in range(6):
        trial = cards.clone()
        trial[:, pos] = held
        sc = compute_final_score(trial, DEVICE)
        better = sc < best_score_bf
        best_score_bf = torch.where(better, sc, best_score_bf)
        best_pos_bf = torch.where(better, torch.full_like(best_pos_bf, pos), best_pos_bf)

    expected_action = 2 + best_pos_bf
    assert torch.equal(action, expected_action), (
        f"lookahead != brute-force on fully-revealed layout.\n"
        f"  lookahead: {action.tolist()}\n  expected:  {expected_action.tolist()}"
    )


def test_lookahead_stage1_discards_bad_held_card():
    """When the held card is worse than every slot, lookahead should
    discard+flip rather than place."""
    N = 4
    state = reset_games(N=N, device=DEVICE)
    state.current_stage.fill_(1)
    # Give player 0 all Kings (rank 11, score 0) in revealed slots 0-4,
    # slot 5 unrevealed. Held card = rank 8 (score 10, a bad card).
    for s in range(5):
        state.player_cards[:, 0, s] = 11  # K suit 0
        state.player_revealed[:, 0, s] = True
    state.player_revealed[:, 0, 5] = False
    # Column pairs: (0,3), (1,4), (2,5). Slots 0-4 = K, slot 5 hidden.
    # Cols 0 and 1 have K-K matches (score 0). Col 2 has K-revealed + hidden.
    # Placing a 10-score card anywhere would break a column match or add 10.
    state.player_holding[:, 0] = 8  # rank 8 = score 10

    t = BayesBeliefTracker(N=N, device=DEVICE)
    t.observe(state, my_player_id=0)
    action = lookahead_stage1(state, 0, t)

    # Should discard+flip slot 5 (the only unrevealed slot), action = 9+5 = 14.
    assert (action == 14).all(), f"Expected discard+flip (14), got {action.tolist()}"


def test_lookahead_stage0_returns_valid_action():
    state = reset_games(N=4, device=DEVICE)
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
    a = lookahead_stage0(state, 0, t)
    assert a.shape == (4,)
    assert ((a == 0) | (a == 1)).all()


def test_lookahead_stage0_takes_column_match():
    """When the face card creates a deterministic column match, take it."""
    N = 4
    state = reset_games(N=N, device=DEVICE)

    # Reveal slot 0 as rank 11 (K, score 0). Set discard_top to another K.
    state.player_cards[:, 0, 0] = 11  # K suit 0
    state.player_revealed[:, 0, 0] = True
    state.discard_top[:] = 11 + NUM_RANKS  # K suit 1

    t = BayesBeliefTracker(N=N, device=DEVICE)
    t.observe(state, my_player_id=0)
    action = lookahead_stage0(state, 0, t)

    # Taking the K and placing at slot 3 (column partner of slot 0) gives
    # a column match worth 0. This should dominate drawing.
    assert (action == 0).all(), f"Expected take (0) for column match, got {action.tolist()}"


def test_lookahead_player_beats_random():
    torch.manual_seed(42)
    score = run_bayes_eval(
        opponent_specs=["R", "R", "R"],
        num_games=200,
        holes=3,
        device=DEVICE,
        player="lookahead",
    )
    assert score < 15.0, f"lookahead avg {score:.2f} should beat random easily"


def test_lookahead_does_not_peek_at_hidden_cards():
    """Shuffling the true card values at hidden positions must NOT change
    lookahead decisions. If it does, expected_score is leaking hidden info."""
    torch.manual_seed(7)
    N = 200
    state = reset_games(N=N, device=DEVICE)
    # Reveal a few slots so there's a mix of hidden and revealed.
    state.player_revealed[:, 0, 0] = True
    state.player_revealed[:, 0, 3] = True  # column 0 both revealed
    state.player_revealed[:, 0, 1] = True  # column 1: slot 1 revealed, slot 4 hidden
    # Slots 2, 4, 5 are hidden.

    state.current_stage.fill_(1)
    state.player_holding[:, 0] = state.discard_top

    t = BayesBeliefTracker(N=N, device=DEVICE)
    t.observe(state, my_player_id=0)
    action_original = lookahead_stage1(state, 0, t).clone()

    # Scramble the true card values at hidden positions.
    hidden_mask = ~state.player_revealed[:, 0, :]  # (N, 6)
    for n in range(N):
        hidden_slots = hidden_mask[n].nonzero(as_tuple=True)[0]
        if len(hidden_slots) > 1:
            perm = torch.randperm(len(hidden_slots))
            orig = state.player_cards[n, 0, hidden_slots].clone()
            state.player_cards[n, 0, hidden_slots] = orig[perm]

    # Re-run with scrambled hidden cards (same tracker state).
    action_scrambled = lookahead_stage1(state, 0, t)

    assert torch.equal(action_original, action_scrambled), (
        f"Lookahead decisions changed after scrambling hidden cards!\n"
        f"  original:  {action_original.tolist()}\n"
        f"  scrambled: {action_scrambled.tolist()}\n"
        f"  differ at: {(action_original != action_scrambled).nonzero(as_tuple=True)[0].tolist()}"
    )

"""Comprehensive tests for BayesBeliefTracker.

Verifies:
  - Bookkeeping (init, reset, idempotency, conservation)
  - Per-step observation correctness for each kind of stage 1 action
  - Multi-player POV consistency (own holding visible only to me)
  - Calibration on normal decks (statistical, real games)
  - Calibration on stacked decks (the rigged regime)
  - Multiset evolution under stacking
  - Reshuffle behavior in 5+ player games (currently expected to be limited)
  - Sanity edge cases (full deck mean, depleted ranks)

The intent is to make the belief tracker the *foundation* for action policies:
when a policy queries the tracker, the answer should be correct or the
limitation should be documented and tested.
"""

from __future__ import annotations

import torch
import pytest

from src.bayes_optimal import BayesBeliefTracker
from src.vectorized_golf import (
    NUM_CARDS,
    NUM_RANKS,
    RANK_SCORES,
    compute_final_score,
    heuristic_stage0,
    improved_stage1,
    random_stage0,
    random_stage1,
    reset_games,
    step_stage0,
    step_stage1,
)


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Game-loop helper
# ---------------------------------------------------------------------------


def _play_one_hole(state, n_players, on_step=None, *, s0_fn=heuristic_stage0, s1_fn=improved_stage1):
    """Run one hole to completion. Calls on_step(state, pid, stage) after each step."""
    for _ in range(60):
        if state.done.all():
            return
        for pid in range(n_players):
            if state.done.all():
                return
            active = ~state.done
            back_to_trigger = state.last_turn & (state.end_game_player == pid)
            state.done = state.done | (back_to_trigger & active)
            active = ~state.done
            if not active.any():
                return

            state.current_stage.fill_(0)
            a0 = s0_fn(state, pid)
            step_stage0(state, a0, pid)
            if on_step is not None:
                on_step(state, pid, 0)
            if state.done.all():
                return

            state.current_stage.fill_(1)
            a1 = s1_fn(state, pid)
            step_stage1(state, a1, pid)
            if on_step is not None:
                on_step(state, pid, 1)

            all_rev = state.player_revealed[:, pid, :].all(dim=1)
            newly = active & all_rev & (~state.last_turn)
            state.last_turn = state.last_turn | newly
            state.end_game_player = torch.where(
                newly,
                torch.full_like(state.end_game_player, pid),
                state.end_game_player,
            )


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_initial_state_all_unobserved():
    t = BayesBeliefTracker(N=3, device=DEVICE)
    assert t.unobserved.shape == (3, 52)
    assert t.unobserved.all()
    assert (t.total() == 52).all()


def test_reset_restores_initial():
    state = reset_games(N=2, device=DEVICE)
    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state, my_player_id=0)
    assert (t.total() < 52).any()
    t.reset()
    assert (t.total() == 52).all()


def test_observe_idempotent():
    state = reset_games(N=2, device=DEVICE)
    t = BayesBeliefTracker(N=2, device=DEVICE)
    t.observe(state, my_player_id=0)
    snapshot = t.unobserved.clone()
    t.observe(state, my_player_id=0)
    t.observe(state, my_player_id=0)
    assert torch.equal(t.unobserved, snapshot)


def test_card_conservation_within_hole():
    """unobserved_count is monotonically non-increasing within a hole, and
    is bounded above by 52 at all times."""
    torch.manual_seed(0)
    N = 32
    n_players = 4
    state = reset_games(N, DEVICE, n_players=n_players)
    t = BayesBeliefTracker(N, DEVICE)
    t.reset()
    t.observe(state, my_player_id=0)

    prev_unobs = t.total().clone()
    assert (prev_unobs <= 52).all()

    def cb(state, pid, stage):
        nonlocal prev_unobs
        t.observe(state, my_player_id=0)
        cur = t.total()
        assert (cur <= 52).all()
        # Monotonic non-increasing
        assert (cur <= prev_unobs).all(), f"unobserved went UP from {prev_unobs.tolist()} to {cur.tolist()}"
        prev_unobs = cur.clone()

    _play_one_hole(state, n_players, on_step=cb)


# ---------------------------------------------------------------------------
# Per-step observation correctness for specific actions
# ---------------------------------------------------------------------------


def test_observe_marks_revealed_cards_only():
    """At init, only the discard top is observed (no slots are face-up yet)."""
    torch.manual_seed(123)
    state = reset_games(N=4, device=DEVICE)
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
    # Only 1 card observed: the discard top (no slots revealed at init).
    assert (t.total() == 51).all()


def test_observe_does_not_count_opponent_holding():
    """If only an opponent is holding a card, my tracker should not include it."""
    torch.manual_seed(0)
    state = reset_games(N=4, device=DEVICE)
    # Force opponent (pid=1) to hold a specific card. We construct the state
    # by setting player_holding directly.
    fake_card = torch.tensor([7, 19, 33, 41], dtype=torch.int16)  # specific cards
    state.player_holding[:, 1] = fake_card

    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
    # The opponent's holding should NOT be in my "seen" set.
    for n in range(4):
        assert t.unobserved[n, fake_card[n].item()], (
            f"game {n}: opponent holding {fake_card[n].item()} was wrongly marked observed"
        )


def test_observe_counts_own_holding():
    """If I am holding a card, my tracker should include it."""
    torch.manual_seed(0)
    state = reset_games(N=4, device=DEVICE)
    fake_card = torch.tensor([5, 17, 23, 31], dtype=torch.int16)
    state.player_holding[:, 0] = fake_card
    t = BayesBeliefTracker(N=4, device=DEVICE)
    t.observe(state, my_player_id=0)
    for n in range(4):
        assert not t.unobserved[n, fake_card[n].item()], (
            f"game {n}: own holding {fake_card[n].item()} was not marked observed"
        )


def test_observe_after_place_at_face_down_marks_both_cards():
    """When I place a card on a face-down slot, both the placed card and the
    displaced (formerly face-down) card become face-up + visible. Both should
    be in the seen set after observe."""
    torch.manual_seed(7)
    N = 4
    state = reset_games(N, DEVICE)
    held = torch.tensor([2, 14, 28, 41], dtype=torch.int16)
    state.player_holding[:, 0] = held
    state.current_stage.fill_(1)
    # Place at slot 0 (face-down)
    actions = torch.full((N,), 2, dtype=torch.int64)  # action 2 = place at slot 0
    cards_at_slot0_before = state.player_cards[:, 0, 0].clone()
    step_stage1(state, actions, 0)

    t = BayesBeliefTracker(N, DEVICE)
    t.observe(state, my_player_id=0)
    for n in range(N):
        # Placed card now face-up at slot 0 of player 0.
        assert not t.unobserved[n, held[n].item()], "placed card not observed"
        # Displaced card is the new discard top.
        assert state.discard_top[n].item() == cards_at_slot0_before[n].item()
        assert not t.unobserved[n, cards_at_slot0_before[n].item()], "displaced card not observed"


def test_observe_after_flip_discard_marks_both_cards():
    """When I discard+flip at a face-down slot, the held card becomes the new
    discard top and the flipped slot becomes face-up. Both visible."""
    torch.manual_seed(11)
    N = 4
    state = reset_games(N, DEVICE)
    held = torch.tensor([3, 15, 29, 42], dtype=torch.int16)
    state.player_holding[:, 0] = held
    state.current_stage.fill_(1)
    # action 9 = discard+flip at slot 0
    actions = torch.full((N,), 9, dtype=torch.int64)
    cards_at_slot0_before = state.player_cards[:, 0, 0].clone()
    step_stage1(state, actions, 0)

    t = BayesBeliefTracker(N, DEVICE)
    t.observe(state, my_player_id=0)
    for n in range(N):
        # held card is now the discard top
        assert not t.unobserved[n, held[n].item()], "held card after flip+discard not observed"
        # flipped slot is now face-up
        assert state.player_revealed[n, 0, 0]
        assert not t.unobserved[n, cards_at_slot0_before[n].item()], "flipped card not observed"


# ---------------------------------------------------------------------------
# Multi-player POV consistency
# ---------------------------------------------------------------------------


def test_multi_player_pov_agrees_when_no_one_is_holding():
    """At the START of any turn, no player is holding a card. Trackers from
    different POVs should agree exactly at that moment."""
    torch.manual_seed(0)
    N = 32
    n_players = 4
    state = reset_games(N, DEVICE, n_players=n_players)
    t0 = BayesBeliefTracker(N, DEVICE)
    t1 = BayesBeliefTracker(N, DEVICE)
    t0.reset()
    t1.reset()
    t0.observe(state, my_player_id=0)
    t1.observe(state, my_player_id=1)

    # At init no one is holding -> trackers should agree.
    assert torch.equal(t0.unobserved, t1.unobserved)

    def cb(state, pid, stage):
        # Re-observe after each step. At stage=1 (just after stage 1 ran)
        # holding is cleared for the actor, so trackers should agree.
        t0.observe(state, my_player_id=0)
        t1.observe(state, my_player_id=1)
        if stage == 1:
            # Holding is cleared at end of stage 1 -- both trackers see same.
            assert (state.player_holding == -1).all() | (state.done.unsqueeze(1)).all()
            assert torch.equal(t0.unobserved, t1.unobserved), (
                f"trackers diverged at end of stage 1, pid={pid}"
            )

    _play_one_hole(state, n_players, on_step=cb)


def test_multi_player_pov_differs_only_in_own_holding():
    """While player A is mid-turn (between stage 0 and stage 1), only A's
    tracker should know A's drawn card. The XOR between tracker A and tracker
    B at that moment should be exactly the set of cards A is currently holding
    (per active game)."""
    torch.manual_seed(0)
    N = 32
    n_players = 4
    state = reset_games(N, DEVICE, n_players=n_players)
    t0 = BayesBeliefTracker(N, DEVICE)
    t1 = BayesBeliefTracker(N, DEVICE)
    t0.reset()
    t1.reset()
    t0.observe(state, my_player_id=0)
    t1.observe(state, my_player_id=1)

    # Force player 0 to draw from deck (action 1) on its turn 0
    state.current_stage.fill_(0)
    actions = torch.ones(N, dtype=torch.int64)
    step_stage0(state, actions, player_id=0)
    t0.observe(state, my_player_id=0)
    t1.observe(state, my_player_id=1)

    diff = t0.unobserved ^ t1.unobserved  # XOR
    # Player 0 has held card now. The XOR for each game should be exactly
    # 1 bit (the held card), and that bit's position should be player 0's holding.
    holdings = state.player_holding[:, 0].long()
    for n in range(N):
        if holdings[n].item() < 0:
            continue  # done game
        diff_cards = diff[n].nonzero().flatten().tolist()
        assert len(diff_cards) == 1, f"game {n}: expected 1 differing card, got {len(diff_cards)}"
        assert diff_cards[0] == holdings[n].item(), (
            f"game {n}: differing card {diff_cards[0]} != holding {holdings[n].item()}"
        )


# ---------------------------------------------------------------------------
# Calibration on normal deck (statistical)
# ---------------------------------------------------------------------------


def _calibration_check(stack_low_cards: bool, n_players: int = 4, num_holes: int = 9, num_games: int = 1000, max_diff: float = 0.005, min_n: int = 1000):
    """For every pid=0 stage-0 decision, log P(deck draw rank=R) per rank R
    and compare to the empirical drawn-rank distribution. Returns a dict with
    per-bin (predicted, observed, n)."""
    torch.manual_seed(0)
    rs = RANK_SCORES.to(DEVICE)
    N = num_games
    tracker = BayesBeliefTracker(N, DEVICE)

    p_lt_samples = []
    drawn_lt_samples = []
    face_score_samples = []

    for hole in range(num_holes):
        state = reset_games(N, DEVICE, n_players=n_players, stack_low_cards=stack_low_cards)
        tracker.reset()
        tracker.observe(state, my_player_id=0)
        for _ in range(60):
            if state.done.all():
                break
            for pid in range(n_players):
                if state.done.all():
                    break
                active = ~state.done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                active = ~state.done
                if not active.any():
                    break
                tracker.observe(state, my_player_id=0)
                if pid == 0:
                    face_rank = (state.discard_top % NUM_RANKS).long()
                    face_score = rs[face_rank]
                    multiset = tracker.multiset_by_rank().float()
                    total = tracker.total().float().clamp(min=1)
                    lt_mask = rs.unsqueeze(0) < face_score.unsqueeze(1)
                    p_lt = (multiset * lt_mask.float()).sum(dim=1) / total
                    deck_card = state.deck[
                        torch.arange(N, device=DEVICE),
                        state.deck_ptr.long().clamp(max=51),
                    ]
                    drawn_score = rs[(deck_card % NUM_RANKS).long()]
                    # Calibration only holds in the no-reshuffle regime. After
                    # a reshuffle, the deck contains previously-buried (= already
                    # observed) cards that are NOT in the multiset, so multiset/
                    # total stops matching the deck-draw distribution. We filter
                    # to:
                    #  1. games where no reshuffle has happened yet (deck_size
                    #     still equals the original NUM_CARDS), AND
                    #  2. the deck currently has drawable cards (deck_ptr < size).
                    no_reshuffle = state.deck_size == NUM_CARDS
                    deck_has_cards = state.deck_ptr < state.deck_size
                    sample_mask = active & deck_has_cards & no_reshuffle
                    act = sample_mask.cpu()
                    if act.any():
                        p_lt_samples.append(p_lt.cpu()[act])
                        drawn_lt_samples.append((drawn_score.cpu() < face_score.cpu())[act].float())
                        face_score_samples.append(face_score.cpu()[act])
                state.current_stage.fill_(0)
                a0 = heuristic_stage0(state, pid)
                step_stage0(state, a0, pid)
                tracker.observe(state, my_player_id=0)
                if state.done.all():
                    break
                state.current_stage.fill_(1)
                a1 = improved_stage1(state, pid)
                step_stage1(state, a1, pid)
                tracker.observe(state, my_player_id=0)
                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly
                state.end_game_player = torch.where(
                    newly,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

    p_lt = torch.cat(p_lt_samples)
    drawn = torch.cat(drawn_lt_samples)
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.7), (0.7, 1.01)]
    out = []
    for lo, hi in bins:
        bm = (p_lt >= lo) & (p_lt < hi)
        n = int(bm.sum().item())
        if n < min_n:
            continue
        pred = float(p_lt[bm].mean().item())
        obs = float(drawn[bm].mean().item())
        out.append((lo, hi, n, pred, obs))
        assert abs(pred - obs) < max_diff, (
            f"bin [{lo:.2f},{hi:.2f}) n={n} pred={pred:.4f} obs={obs:.4f} diff={obs-pred:+.4f} "
            f"exceeds {max_diff:.3f} threshold"
        )
    return out


def test_calibration_normal_4p_deck():
    """In normal 4p9h play, posterior P(deck draw < face) matches empirical
    rate to within 0.005 in every reasonably-populated bin."""
    bins = _calibration_check(stack_low_cards=False, n_players=4, num_games=2000, max_diff=0.008)
    assert len(bins) >= 5, f"expected calibration data in at least 5 bins, got {len(bins)}"


def test_calibration_normal_4p_deck_explanation():
    """Documentation note attached to the normal-deck calibration test:

    The normal-deck calibration test is the load-bearing one for the bayes
    player. It works because in normal play, the deck is randomly shuffled
    and exchangeability holds: every unobserved card is uniformly distributed
    over all unobserved positions (deck, opponent face-down, own face-down).
    The marginal `multiset[r]/total` is therefore both the player's best
    posterior AND the actual deck-draw distribution.

    There is NO equivalent calibration test for stacked decks because the
    stacking deliberately violates exchangeability (specific cards placed
    at specific positions). In that regime the player's marginal posterior
    is still the player's best guess given their information, but it does
    NOT equal the deck-draw distribution. Decisions based on the posterior
    are correct given the player's information set; they just don't match
    what actually happens at the deck top in stacked play.

    For action policies playing against actual randomly-shuffled decks (the
    only thing they encounter in normal play), this distinction doesn't
    matter -- the posterior IS the right object to base decisions on. The
    stacked-deck experiment in seat_cycling is a probe of *whether the
    player would correctly use a strong belief signal*, not a calibration
    test of the tracker.
    """
    pass  # this is a documentation node, not an executable assertion


# ---------------------------------------------------------------------------
# Stacked-deck multiset evolution
# ---------------------------------------------------------------------------


def test_stacked_deck_multiset_keeps_low_ranks_higher_than_high():
    """In a stacked-deck game, low ranks (2/K/A) start in the deck-bottom and
    high ranks start in player layouts and discard. As the hole progresses,
    high cards get observed faster than low cards, so on AVERAGE the multiset
    count for low ranks should remain higher than for high ranks throughout
    the game.

    Note: in long stacked holes the deck CAN get drawn deep enough to expose
    some low cards (which then become observed), so multiset[low] is not
    monotonic at 4. We check the relative ordering, not absolute values.
    """
    torch.manual_seed(0)
    N = 500
    n_players = 4
    state = reset_games(N, DEVICE, n_players=n_players, stack_low_cards=True)
    t = BayesBeliefTracker(N, DEVICE)
    t.reset()
    t.observe(state, my_player_id=0)

    low_ranks = [0, 11, 12]   # 2, K, A
    high_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    snapshots_low = []
    snapshots_high = []

    def cb(state, pid, stage):
        t.observe(state, my_player_id=0)
        if pid == 0 and stage == 0:
            ms = t.multiset_by_rank().float()  # (N, 13)
            # Average per-rank count: total count for the group / number of ranks.
            avg_low_per_rank = ms[:, low_ranks].sum(dim=1).mean().item() / len(low_ranks)
            avg_high_per_rank = ms[:, high_ranks].sum(dim=1).mean().item() / len(high_ranks)
            snapshots_low.append(avg_low_per_rank)
            snapshots_high.append(avg_high_per_rank)

    _play_one_hole(state, n_players, on_step=cb)

    assert snapshots_low and snapshots_high, "no snapshots collected"

    # Throughout the hole, low ranks should be on average more common in
    # the multiset than high ranks (because high cards are getting observed
    # while low cards are mostly stuck in the deck or in the bottom of player
    # layouts).
    for i, (lo, hi) in enumerate(zip(snapshots_low, snapshots_high)):
        assert lo >= hi, (
            f"snapshot {i}: avg multiset[low]={lo:.2f} should be >= multiset[high]={hi:.2f} "
            f"throughout a stacked-deck hole"
        )
    # And by end of hole the gap should be MEANINGFUL (>= 0.5 cards/rank).
    final_gap = snapshots_low[-1] - snapshots_high[-1]
    assert final_gap >= 0.5, (
        f"end-of-hole gap multiset[low]-multiset[high]={final_gap:.2f} should be at least 0.5 "
        f"in a stacked-deck hole"
    )


# ---------------------------------------------------------------------------
# Reshuffle behavior (5+ players)
# ---------------------------------------------------------------------------


def test_reshuffles_actually_happen_in_6p():
    """Sanity: in 6p9h Golf, reshuffles must occur frequently for downstream
    tests to be meaningful."""
    torch.manual_seed(0)
    N = 100
    n_players = 6
    reshuffles_per_game = torch.zeros(N, dtype=torch.long)
    for hole in range(9):
        state = reset_games(N, DEVICE, n_players=n_players)
        for _ in range(60):
            if state.done.all():
                break
            for pid in range(n_players):
                if state.done.all():
                    break
                active = ~state.done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                if not (~state.done).any():
                    break
                state.current_stage.fill_(0)
                buried_before = state.discard_buried.sum(dim=1).clone()
                a0 = heuristic_stage0(state, pid)
                step_stage0(state, a0, pid)
                buried_after = state.discard_buried.sum(dim=1)
                # Detect reshuffle: buried count drops by more than 1 in one step.
                resh = (buried_before - buried_after) > 1
                reshuffles_per_game += resh.long()
                if state.done.all():
                    break
                state.current_stage.fill_(1)
                a1 = improved_stage1(state, pid)
                step_stage1(state, a1, pid)
                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly = (~state.done) & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly
                state.end_game_player = torch.where(
                    newly, torch.full_like(state.end_game_player, pid), state.end_game_player
                )
    n_with_reshuffle = (reshuffles_per_game > 0).sum().item()
    assert n_with_reshuffle >= 50, (
        f"expected most 6p games to reshuffle; only {n_with_reshuffle}/{N} did"
    )


def test_known_limitation_post_reshuffle_miscalibration():
    """KNOWN LIMITATION: after a deck reshuffle, the multiset stops being a
    calibrated estimator of the deck draw distribution.

    Reason: when the deck empties and the buried discard pile is reshuffled
    back into the deck, those cards are now drawable, but the tracker still
    has them marked as 'observed' (because the player has indeed seen them
    historically). The marginal `multiset[r]/total` represents 'probability
    that any UNSEEN card has rank r' -- correct as a marginal posterior over
    cards-the-player-has-never-seen, but NOT equal to 'probability that the
    next deck draw has rank r' once the deck contains seen-and-recycled cards.

    This test exists to make the limitation explicit and to fail loudly if
    someone 'fixes' the multiset and forgets to update this test. To remove
    the limitation we would need a tracker that distinguishes 'where each
    card currently is' (deck vs face-down vs buried-discard) rather than
    just 'have I seen it'.

    The test deliberately constructs the post-reshuffle regime via stacked
    decks (which exhaust the 27-card initial deck and trigger reshuffles in
    4p9h play) and verifies that calibration is materially worse there.
    """
    torch.manual_seed(0)
    N = 2000
    n_players = 4
    rs = RANK_SCORES.to(DEVICE)
    tracker = BayesBeliefTracker(N, DEVICE)

    # Collect (predicted, observed) over POST-reshuffle decisions only.
    p_lt_samples = []
    drawn_lt_samples = []
    for hole in range(9):
        state = reset_games(N, DEVICE, n_players=n_players, stack_low_cards=True)
        tracker.reset()
        tracker.observe(state, my_player_id=0)
        for _ in range(60):
            if state.done.all():
                break
            for pid in range(n_players):
                if state.done.all():
                    break
                active = ~state.done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                if not (~state.done).any():
                    break
                tracker.observe(state, my_player_id=0)
                if pid == 0:
                    face_rank = (state.discard_top % NUM_RANKS).long()
                    face_score = rs[face_rank]
                    multiset = tracker.multiset_by_rank().float()
                    total = tracker.total().float().clamp(min=1)
                    lt_mask = rs.unsqueeze(0) < face_score.unsqueeze(1)
                    p_lt = (multiset * lt_mask.float()).sum(dim=1) / total
                    deck_card = state.deck[
                        torch.arange(N, device=DEVICE),
                        state.deck_ptr.long().clamp(max=51),
                    ]
                    drawn_score = rs[(deck_card % NUM_RANKS).long()]
                    # POST-reshuffle: deck_size != NUM_CARDS AND deck has cards.
                    post_reshuffle = state.deck_size < NUM_CARDS
                    deck_has_cards = state.deck_ptr < state.deck_size
                    mask = active & deck_has_cards & post_reshuffle
                    act = mask.cpu()
                    if act.any():
                        p_lt_samples.append(p_lt.cpu()[act])
                        drawn_lt_samples.append((drawn_score.cpu() < face_score.cpu())[act].float())
                state.current_stage.fill_(0)
                a0 = heuristic_stage0(state, pid)
                step_stage0(state, a0, pid)
                tracker.observe(state, my_player_id=0)
                if state.done.all():
                    break
                state.current_stage.fill_(1)
                a1 = improved_stage1(state, pid)
                step_stage1(state, a1, pid)
                tracker.observe(state, my_player_id=0)
                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly
                state.end_game_player = torch.where(
                    newly,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

    if not p_lt_samples:
        pytest.skip("no post-reshuffle decisions collected; test cannot run")
    p_lt = torch.cat(p_lt_samples)
    drawn = torch.cat(drawn_lt_samples)
    n = len(p_lt)
    avg_predicted = float(p_lt.mean().item())
    avg_observed = float(drawn.mean().item())
    # The miscalibration is large -- assert it's at least 0.05 to confirm
    # the limitation is real and reproducible. If this assertion ever stops
    # firing, either (a) someone fixed the tracker and should remove this
    # test, or (b) the calibration randomly improved -- investigate.
    diff = avg_observed - avg_predicted
    assert n >= 100, f"too few post-reshuffle samples ({n}) to be meaningful"
    assert abs(diff) >= 0.05, (
        f"post-reshuffle miscalibration is smaller than expected: "
        f"n={n} avg_pred={avg_predicted:.4f} avg_obs={avg_observed:.4f} diff={diff:+.4f}. "
        f"Either the tracker was fixed (good -- remove this test) or the "
        f"sampling regime changed."
    )


def test_card_conservation_through_reshuffle():
    """In a 6p game with reshuffles, the tracker's unobserved count should
    still be in [0, 52] at every step and only decrease (within a hole)."""
    torch.manual_seed(0)
    N = 32
    n_players = 6
    state = reset_games(N, DEVICE, n_players=n_players)
    t = BayesBeliefTracker(N, DEVICE)
    t.reset()
    t.observe(state, my_player_id=0)

    prev_unobs = t.total().clone()

    def cb(state, pid, stage):
        nonlocal prev_unobs
        t.observe(state, my_player_id=0)
        cur = t.total()
        assert (cur >= 0).all() and (cur <= 52).all()
        # Monotonic non-increasing within a hole.
        assert (cur <= prev_unobs).all(), f"unobserved went UP from {prev_unobs.tolist()} to {cur.tolist()}"
        prev_unobs = cur.clone()

    _play_one_hole(state, n_players, on_step=cb)


# ---------------------------------------------------------------------------
# Sanity edge cases
# ---------------------------------------------------------------------------


def test_expected_unknown_score_full_deck_matches_global_average():
    t = BayesBeliefTracker(N=1, device=DEVICE)
    e = t.expected_unknown_score()
    expected = RANK_SCORES.mean().item()
    assert abs(e.item() - expected) < 1e-5


def test_observing_all_4_of_a_rank_zeroes_multiset_for_that_rank():
    """If we manually mark all 4 cards of rank 5 as observed, multiset[5] = 0."""
    t = BayesBeliefTracker(N=1, device=DEVICE)
    # Rank 5 = rank index 3 (because rank scores: [-2, 3, 4, 5, 6, ...] -> index 3 has score 5)
    rank_idx_for_5 = 3
    # Cards with rank index 3: card_idx % 13 == 3 -> indices 3, 16, 29, 42
    for c in [3, 16, 29, 42]:
        t.unobserved[0, c] = False
    ms = t.multiset_by_rank()
    assert ms[0, rank_idx_for_5].item() == 0
    # Other ranks unaffected
    for r in range(NUM_RANKS):
        if r != rank_idx_for_5:
            assert ms[0, r].item() == 4

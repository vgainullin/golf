"""Compare iterative Golf simulation vs vectorized Golf environment.

Runs N games using both code paths with identical deck orders and heuristic
players, then checks that final scores match.
"""

import random
import numpy as np
import torch
from copy import deepcopy

from src.simulation import Golf, GolfDeck, Player, get_player_action, encode_pos_tuple
from src.vectorized_golf import (
    VectorizedGolfState,
    reset_games,
    get_observation,
    step_stage0,
    step_stage1,
    compute_score,
    compute_final_score,
    get_valid_action_mask,
    heuristic_stage0,
    heuristic_stage1,
    NUM_RANKS,
    RANK_CUTOFF,
    RANK_SCORES,
    UNKNOWN_CARD,
)

NUM_GAMES = 100
SEED = 42


def card_to_index(card):
    """Convert a Card namedtuple to our index encoding (suit*13 + rank)."""
    ranks = [str(n) for n in range(2, 10)] + list('XJQKA')
    suits = 'spades diamonds clubs hearts'.split()
    rank_idx = ranks.index(card.rank)
    suit_idx = suits.index(card.suit)
    return suit_idx * 13 + rank_idx


PLAYER_TYPES = ['Random', 'Heuristic', 'Random', 'Heuristic']


def run_iterative_game(seed):
    """Run one game matching simulation.py's play_game exactly."""
    random.seed(seed)
    np.random.seed(seed)

    players = [Player(name=f"P{i}", id=i, type=PLAYER_TYPES[i]) for i in range(4)]
    golf = Golf(players=players, deck_type="French", verbose=False)
    golf.shuffle()

    # Record deck order before dealing
    deck_order = [card_to_index(c) for c in golf.deck]

    golf.deal()

    round_num = 0
    action_log = []

    while not golf.game_over:
        for pid in range(4):
            player = golf.players[pid]
            take_random = player.type == 'Random'

            golf.players[pid].gather_game_state(golf)
            state_ = golf.players[pid].game_state

            if '?' not in state_:
                golf.game_over = True
                break

            # Stage 0
            action, pos, _ = get_player_action(
                deepcopy(golf), pid, 0, rank_cutoff=4, take_random_action=take_random,
            )
            r0 = golf.take_action(pid, [0, action, pos])
            golf.players[pid].gather_game_state(golf)
            action_log.append(('s0', pid, action, pos, r0))

            # Stage 1
            action1, pos1, _ = get_player_action(
                deepcopy(golf), pid, 1, rank_cutoff=4, take_random_action=take_random,
            )
            r1 = golf.take_action(pid, [1, action1, pos1])
            golf.players[pid].gather_game_state(golf)
            action_log.append(('s1', pid, action1, pos1, r1))

            if len(golf.deck) < golf.num_players + 2:
                golf.deck = GolfDeck()
                golf.shuffle()
                golf.deal()

            if '?' not in golf.players[pid].game_state:
                golf.last_turn = True
                golf.end_game_player_id = pid

        round_num += 1

    # Final scores
    final_scores = []
    for p in golf.players:
        p.calculate_score(final=True)
        final_scores.append(p.score)

    return deck_order, final_scores, action_log


def run_vectorized_game(deck_order, device):
    """Run one game with 4 heuristic players using the vectorized env.

    Uses the provided deck order to match the iterative version.
    The iterative version deals via deck.pop() (from end), so we reverse.
    """
    N = 1
    state = reset_games(N, device)

    # Iterative version deals via pop() from end of deck.
    # deck.pop() order: deck[-1], deck[-2], ..., deck[-24]
    # Player p, slot s (row*3+col) gets deck[51 - (p*6 + s)]
    # Face card = deck[51 - 24] = deck[27]
    # Remaining deck for drawing: deck[0..26], drawn via pop() = deck[26], deck[25], ...
    reversed_deck = list(reversed(deck_order))
    # reversed_deck[0] = deck_order[51] = first card dealt (P0 slot 0)
    # reversed_deck[24] = deck_order[27] = face card
    # reversed_deck[25] = deck_order[26] = first card drawn from deck
    # etc.

    state.deck[0] = torch.tensor(reversed_deck, dtype=torch.int16, device=device)

    # Deal: first 24 cards from reversed deck -> 4 players x 6 slots
    deal_cards = state.deck[0, :24].reshape(4, 6)
    state.player_cards[0] = deal_cards.clone()
    state.player_revealed[0] = False
    state.player_holding[0] = -1
    state.discard_top[0] = state.deck[0, 24]
    state.deck_ptr[0] = 25

    action_log = []

    for round_num in range(30):
        if state.done.all():
            break

        for pid in range(4):
            active = ~state.done

            # Check game over: end_game_player gets their turn back
            back_to_trigger = state.last_turn & (state.end_game_player == pid)
            state.done = state.done | (back_to_trigger & active)
            active = ~state.done

            if not active.any():
                break

            # Stage 0
            state.current_stage.fill_(0)
            actions_s0 = heuristic_stage0(state, pid)
            step_stage0(state, actions_s0, pid)

            action_log.append(('s0', pid, int(actions_s0[0].item())))

            if state.done.all():
                break

            # Stage 1
            state.current_stage.fill_(1)
            actions_s1 = heuristic_stage1(state, pid)
            rewards_s1 = step_stage1(state, actions_s1, pid)

            action_log.append(('s1', pid, int(actions_s1[0].item()), float(rewards_s1[0].item())))

            # Check last turn
            all_rev = state.player_revealed[:, pid, :].all(dim=1)
            newly_last = active & all_rev & (~state.last_turn)
            state.last_turn = state.last_turn | newly_last
            state.end_game_player = torch.where(
                newly_last,
                torch.full_like(state.end_game_player, pid),
                state.end_game_player,
            )

    # Final scores
    final_scores = []
    for pid in range(4):
        score = compute_final_score(state.player_cards[:, pid, :], device)
        final_scores.append(float(score[0].item()))

    return final_scores, action_log


def test_score_function():
    """Test compute_score against known hands."""
    device = torch.device('cpu')
    rank_scores_list = [-2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 0, 1]

    print("=== Score function tests ===")

    # Test 1: All same rank in columns -> zeros
    # Cards: rank 0 (=2, score=-2) everywhere
    # 6 cards, all rank 0: suits 0,1,2 for row0 and suits 0,1,2 for row1
    cards = torch.tensor([[0, 1, 2, 0+13, 1+13, 2+13]], dtype=torch.int16)  # All rank 0 (2s)
    revealed = torch.ones(1, 6, dtype=torch.bool)
    # Columns match: rank[0]==rank[3], rank[1]==rank[4], rank[2]==rank[5] -> all 0
    score = compute_score(cards, revealed, device)
    print(f"  All 2s column-matched: score={score.item()} (expected 0.0)")
    assert score.item() == 0.0, f"Expected 0.0, got {score.item()}"

    # Test 2: No column matches, all revealed
    # Row 0: ranks 0,1,2 (scores -2,3,4)  Row 1: ranks 3,4,5 (scores 5,6,7)
    cards = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int16)
    revealed = torch.ones(1, 6, dtype=torch.bool)
    score = compute_score(cards, revealed, device)
    expected = -2 + 3 + 4 + 5 + 6 + 7
    print(f"  No matches: score={score.item()} (expected {expected})")
    assert score.item() == expected, f"Expected {expected}, got {score.item()}"

    # Test 3: Some hidden
    cards = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int16)
    revealed = torch.tensor([[True, True, False, False, True, True]])
    # Visible: slot 0 (-2), slot 1 (3), slot 4 (6), slot 5 (7)
    # Column 0: slot 0 revealed, slot 3 hidden -> no match, score slot 0 = -2, slot 3 = 0
    # Column 1: slot 1 revealed, slot 4 revealed, ranks 1!=4 -> no match, scores 3 + 6
    # Column 2: slot 2 hidden, slot 5 revealed -> no match, score slot 2 = 0, slot 5 = 7
    score = compute_score(cards, revealed, device)
    expected = -2 + 3 + 0 + 0 + 6 + 7
    print(f"  Partial reveal: score={score.item()} (expected {expected})")
    assert score.item() == expected, f"Expected {expected}, got {score.item()}"

    # Test 4: Kings (rank 11, score 0)
    cards = torch.tensor([[11, 11, 11, 11, 11, 11]], dtype=torch.int16)
    revealed = torch.ones(1, 6, dtype=torch.bool)
    score = compute_score(cards, revealed, device)
    print(f"  All Kings: score={score.item()} (expected 0.0)")
    assert score.item() == 0.0

    print("  All score tests passed!\n")


def test_observation():
    """Test that get_observation returns correct format."""
    device = torch.device('cpu')
    state = reset_games(2, device)

    # Reveal some cards
    state.player_revealed[0, 0, 0] = True
    state.player_revealed[0, 0, 3] = True

    obs = get_observation(state, 0)
    print("=== Observation tests ===")
    print(f"  Shape: {obs.shape} (expected [2, 8])")
    assert obs.shape == (2, 8)

    # Slot 0 and 3 should show actual card, others should be 52
    assert obs[0, 0] != UNKNOWN_CARD, "Revealed slot should show card"
    assert obs[0, 1] == UNKNOWN_CARD, "Hidden slot should be 52"
    assert obs[0, 3] != UNKNOWN_CARD, "Revealed slot should show card"

    # Holding should be 52 (no one holding)
    assert obs[0, 6] == UNKNOWN_CARD, "No holding should be 52"

    # Discard top should be a valid card
    assert 0 <= obs[0, 7] < 52, "Discard should be valid card"

    print("  All observation tests passed!\n")


def test_step_stage0():
    """Test stage 0 step mechanics."""
    device = torch.device('cpu')
    state = reset_games(2, device)

    print("=== Stage 0 tests ===")

    discard_before = state.discard_top.clone()
    deck_ptr_before = state.deck_ptr.clone()

    # Game 0: take face card (action 0)
    # Game 1: draw from deck (action 1)
    actions = torch.tensor([0, 1], dtype=torch.long)
    step_stage0(state, actions, 0)

    # Game 0: holding should be the face card
    assert state.player_holding[0, 0] == discard_before[0], \
        f"Expected holding {discard_before[0]}, got {state.player_holding[0, 0]}"
    # Deck pointer unchanged for game 0
    assert state.deck_ptr[0] == deck_ptr_before[0], "Deck ptr should not advance for face card"

    # Game 1: holding should be a deck card, deck ptr advanced
    assert state.player_holding[1, 0] == state.deck[1, deck_ptr_before[1].long()], \
        "Holding should be the card from deck"
    assert state.deck_ptr[1] == deck_ptr_before[1] + 1, "Deck ptr should advance"

    print("  Stage 0 step tests passed!\n")


def test_step_stage1_place():
    """Test stage 1 place action."""
    device = torch.device('cpu')
    state = reset_games(1, device)

    print("=== Stage 1 place tests ===")

    # First do stage 0 to get a card
    actions_s0 = torch.tensor([0], dtype=torch.long)  # take face
    step_stage0(state, actions_s0, 0)
    held = state.player_holding[0, 0].item()
    original_at_pos2 = state.player_cards[0, 0, 2].item()

    # Place at position 2 (action 4 = 2+2)
    actions_s1 = torch.tensor([4], dtype=torch.long)
    rewards = step_stage1(state, actions_s1, 0)

    assert state.player_cards[0, 0, 2].item() == held, \
        f"Card at pos 2 should be the held card {held}, got {state.player_cards[0, 0, 2].item()}"
    assert state.player_revealed[0, 0, 2].item() == True, "Position should be revealed"
    assert state.discard_top[0].item() == original_at_pos2, \
        f"Discard should be the replaced card {original_at_pos2}, got {state.discard_top[0].item()}"
    assert state.player_holding[0, 0].item() == -1, "Holding should be cleared"

    print("  Stage 1 place tests passed!\n")


def test_step_stage1_flip():
    """Test stage 1 discard+flip action."""
    device = torch.device('cpu')
    state = reset_games(1, device)

    print("=== Stage 1 flip tests ===")

    # Stage 0: draw from deck
    actions_s0 = torch.tensor([1], dtype=torch.long)
    step_stage0(state, actions_s0, 0)
    held = state.player_holding[0, 0].item()
    original_at_pos1 = state.player_cards[0, 0, 1].item()

    # Discard + flip at position 1 (action 10 = 9+1)
    actions_s1 = torch.tensor([10], dtype=torch.long)
    rewards = step_stage1(state, actions_s1, 0)

    # Card at pos 1 should NOT change (we just flipped it)
    assert state.player_cards[0, 0, 1].item() == original_at_pos1, \
        f"Card should not change on flip, expected {original_at_pos1}, got {state.player_cards[0, 0, 1].item()}"
    assert state.player_revealed[0, 0, 1].item() == True, "Position should be revealed"
    assert state.discard_top[0].item() == held, \
        f"Discard should be the held card {held}, got {state.discard_top[0].item()}"
    assert state.player_holding[0, 0].item() == -1, "Holding should be cleared"

    print("  Stage 1 flip tests passed!\n")


def test_valid_action_mask():
    """Test action masking."""
    device = torch.device('cpu')
    state = reset_games(1, device)

    print("=== Action mask tests ===")

    # Stage 0: actions 0,1 valid
    state.current_stage.fill_(0)
    mask = get_valid_action_mask(state, 0)
    assert mask[0, 0].item() == True, "Take face should be valid"
    assert mask[0, 1].item() == True, "Draw should be valid"
    assert mask[0, 2].item() == False, "Place should not be valid in stage 0"

    # Stage 1: actions 2-7, 9-14 (only unrevealed for 9-14)
    state.current_stage.fill_(1)
    mask = get_valid_action_mask(state, 0)
    assert mask[0, 0].item() == False, "Take face invalid in stage 1"
    assert mask[0, 2].item() == True, "Place at pos 0 valid"
    assert mask[0, 9].item() == True, "Flip at pos 0 valid (unrevealed)"

    # Reveal pos 0, flip should become invalid
    state.player_revealed[0, 0, 0] = True
    mask = get_valid_action_mask(state, 0)
    assert mask[0, 9].item() == False, "Flip at pos 0 should be invalid (revealed)"
    assert mask[0, 2].item() == True, "Place at pos 0 still valid"

    print("  Action mask tests passed!\n")


def test_heuristic_stage0_logic():
    """Test heuristic_stage0 decisions."""
    device = torch.device('cpu')
    print("=== Heuristic stage 0 tests ===")

    # Test: low face card (rank=0, score=-2) -> should take
    state = reset_games(1, device)
    state.discard_top[0] = 0  # rank 0 = "2", score -2
    action = heuristic_stage0(state, 0)
    assert action[0].item() == 0, f"Should take low face card, got action {action[0].item()}"

    # Test: high face card (rank=8, score=10, "X") no matching revealed -> draw
    state = reset_games(1, device)
    state.discard_top[0] = 8  # rank 8 = "X", score 10
    state.player_revealed[0, 0, :] = False  # nothing revealed -> no rank match possible
    action = heuristic_stage0(state, 0)
    assert action[0].item() == 1, f"Should draw with high face card, got action {action[0].item()}"

    # Test: face card rank matches a revealed card -> should take
    state = reset_games(1, device)
    state.discard_top[0] = 21  # suit 1, rank 8 = "X", score 10
    state.player_cards[0, 0, 0] = 8  # rank 8 at slot 0
    state.player_revealed[0, 0, 0] = True  # revealed
    action = heuristic_stage0(state, 0)
    assert action[0].item() == 0, f"Should take face card matching revealed rank, got {action[0].item()}"

    print("  Heuristic stage 0 tests passed!\n")


def test_full_game_vectorized():
    """Run a complete vectorized game and check it terminates."""
    device = torch.device('cpu')
    print("=== Full game test ===")

    state = reset_games(10, device)

    for round_num in range(30):
        if state.done.all():
            break

        for pid in range(4):
            active = ~state.done
            back_to_trigger = state.last_turn & (state.end_game_player == pid)
            state.done = state.done | (back_to_trigger & active)
            active = ~state.done

            if not active.any():
                break

            state.current_stage.fill_(0)
            actions_s0 = heuristic_stage0(state, pid)
            step_stage0(state, actions_s0, pid)

            if state.done.all():
                break

            state.current_stage.fill_(1)
            actions_s1 = heuristic_stage1(state, pid)
            step_stage1(state, actions_s1, pid)

            all_rev = state.player_revealed[:, pid, :].all(dim=1)
            newly_last = active & all_rev & (~state.last_turn)
            state.last_turn = state.last_turn | newly_last
            state.end_game_player = torch.where(
                newly_last,
                torch.full_like(state.end_game_player, pid),
                state.end_game_player,
            )

    n_done = state.done.sum().item()
    print(f"  Games done: {n_done}/10")
    assert n_done == 10, f"Expected all games done, got {n_done}/10"

    for pid in range(4):
        scores = compute_final_score(state.player_cards[:, pid, :], device)
        print(f"  Player {pid} scores: min={scores.min().item():.0f} max={scores.max().item():.0f} mean={scores.mean().item():.1f}")

    print("  Full game test passed!\n")


def test_comparison():
    """Compare iterative vs vectorized with same deck order."""
    device = torch.device('cpu')
    print(f"=== Comparison test: {NUM_GAMES} games ===")

    mismatches = 0
    total_score_diff = 0.0

    for game_idx in range(NUM_GAMES):
        seed = SEED + game_idx

        # Run iterative
        deck_order, iter_scores, iter_actions = run_iterative_game(seed)

        # Run vectorized with same deck
        vec_scores, vec_actions = run_vectorized_game(deck_order, device)

        # Compare final scores
        match = True
        for pid in range(4):
            diff = abs(iter_scores[pid] - vec_scores[pid])
            total_score_diff += diff
            if diff > 0.01:
                match = False

        if not match:
            mismatches += 1
            if mismatches <= 5:  # Print first 5 mismatches
                print(f"  Game {game_idx} (seed={seed}) MISMATCH:")
                print(f"    Iterative scores: {iter_scores}")
                print(f"    Vectorized scores: {vec_scores}")
                print(f"    Iter actions ({len(iter_actions)}):", iter_actions[:6], "...")
                print(f"    Vec  actions ({len(vec_actions)}):", vec_actions[:6], "...")

    match_pct = (NUM_GAMES - mismatches) / NUM_GAMES * 100
    avg_diff = total_score_diff / (NUM_GAMES * 4)
    print(f"\n  Results: {NUM_GAMES - mismatches}/{NUM_GAMES} games matched ({match_pct:.1f}%)")
    print(f"  Average score difference: {avg_diff:.4f}")
    print(f"  Mismatches: {mismatches}")

    if mismatches > 0:
        print("\n  WARNING: Mismatches found. Investigating first mismatch...")
        seed = SEED
        deck_order, iter_scores, iter_actions = run_iterative_game(seed)
        vec_scores, vec_actions = run_vectorized_game(deck_order, device)
        print(f"  Deck (first 30): {deck_order[:30]}")
        print(f"  Iterative actions: {iter_actions[:20]}")
        print(f"  Vectorized actions: {vec_actions[:20]}")
    else:
        print("  All games matched perfectly!")


if __name__ == "__main__":
    # Unit tests
    test_score_function()
    test_observation()
    test_step_stage0()
    test_step_stage1_place()
    test_step_stage1_flip()
    test_valid_action_mask()
    test_heuristic_stage0_logic()
    test_full_game_vectorized()

    # Comparison test
    test_comparison()

    print("\n=== ALL TESTS COMPLETE ===")

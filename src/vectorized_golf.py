"""Vectorized Golf environment for GPU-accelerated batched play.

Represents N games simultaneously as tensors, enabling batched model
inference and eliminating Python loop overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# Card encoding: suit_idx * 13 + rank_idx (0-51), 52 = unknown
UNKNOWN_CARD = 52
NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52
NUM_ACTIONS = 16

# Rank indices: 0=2, 1=3, ..., 7=9, 8=X, 9=J, 10=Q, 11=K, 12=A
# Scores:       -2,  3,  ..., 9,   10,  10,  10,  0,   1
RANK_SCORES = torch.tensor([-2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 0, 1], dtype=torch.float32)

RANK_CUTOFF = 4


@dataclass
class VectorizedGolfState:
    """All game state as tensors, first dim = N (batch of games)."""
    player_cards: torch.Tensor      # (N, 4, 6) int16 -- card indices
    player_revealed: torch.Tensor   # (N, 4, 6) bool -- which slots face-up
    player_holding: torch.Tensor    # (N, 4) int16 -- held card, -1 if none
    deck: torch.Tensor              # (N, 52) int16 -- pre-shuffled full deck
    deck_ptr: torch.Tensor          # (N,) int32 -- next card to draw
    discard_top: torch.Tensor       # (N,) int16 -- top of discard pile
    current_player: torch.Tensor    # (N,) int8 -- whose turn (0-3)
    current_stage: torch.Tensor     # (N,) int8 -- 0 or 1
    last_turn: torch.Tensor         # (N,) bool -- end-game triggered
    end_game_player: torch.Tensor   # (N,) int8 -- who triggered (-1 if not)
    done: torch.Tensor              # (N,) bool -- game finished
    scores: torch.Tensor            # (N, 4) float32 -- cumulative scores


def reset_games(N: int, device: torch.device) -> VectorizedGolfState:
    """Create N fresh games: shuffle decks, deal 24 cards, set face card."""
    # Generate shuffled decks: each row is a permutation of 0..51
    deck = torch.stack([torch.randperm(NUM_CARDS, device=device) for _ in range(N)])
    deck = deck.to(torch.int16)

    # Deal 24 cards: 6 per player x 4 players
    # Player p gets cards at deck positions [p*6 .. p*6+5]
    # Layout: player_cards[n, p, slot] where slot is 0..5 (row0: 0,1,2  row1: 3,4,5)
    deal_cards = deck[:, :24].reshape(N, 4, 6)
    player_cards = deal_cards.clone()

    # All cards start face-down
    player_revealed = torch.zeros(N, 4, 6, dtype=torch.bool, device=device)

    # No one is holding a card
    player_holding = torch.full((N, 4), -1, dtype=torch.int16, device=device)

    # Face card is the 25th card (index 24)
    discard_top = deck[:, 24].clone()

    # Deck pointer starts at 25 (cards 0-23 dealt, 24 is face card)
    deck_ptr = torch.full((N,), 25, dtype=torch.int32, device=device)

    return VectorizedGolfState(
        player_cards=player_cards,
        player_revealed=player_revealed,
        player_holding=player_holding,
        deck=deck,
        deck_ptr=deck_ptr,
        discard_top=discard_top,
        current_player=torch.zeros(N, dtype=torch.int8, device=device),
        current_stage=torch.zeros(N, dtype=torch.int8, device=device),
        last_turn=torch.zeros(N, dtype=torch.bool, device=device),
        end_game_player=torch.full((N,), -1, dtype=torch.int8, device=device),
        done=torch.zeros(N, dtype=torch.bool, device=device),
        scores=torch.zeros(N, 4, dtype=torch.float32, device=device),
    )


def card_rank(card_indices: torch.Tensor) -> torch.Tensor:
    """Extract rank index (0-12) from card index (0-51). -1 for invalid."""
    return torch.where(card_indices >= 0, card_indices % NUM_RANKS, torch.tensor(-1, device=card_indices.device))


def compute_score(cards: torch.Tensor, revealed: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute score for a player's 6-card layout.

    Args:
        cards: (N, 6) int16 card indices
        revealed: (N, 6) bool which are face-up
        device: torch device

    Returns:
        (N,) float32 scores
    """
    rank_scores = RANK_SCORES.to(device)
    ranks = cards % NUM_RANKS  # (N, 6)

    # Get per-card scores
    card_scores = rank_scores[ranks.long()]  # (N, 6)

    # Zero out unrevealed cards (they don't count toward visible score)
    card_scores = card_scores * revealed.float()

    # Column matching: cards are laid out as 2 rows x 3 cols
    # slot 0,1,2 = row 0; slot 3,4,5 = row 1
    # Column match: if rank[slot_i] == rank[slot_i+3] and both revealed, both score 0
    for col in range(3):
        row0_slot = col
        row1_slot = col + 3
        both_revealed = revealed[:, row0_slot] & revealed[:, row1_slot]
        ranks_match = ranks[:, row0_slot] == ranks[:, row1_slot]
        zero_mask = both_revealed & ranks_match
        card_scores[:, row0_slot] = torch.where(zero_mask, torch.zeros_like(card_scores[:, row0_slot]), card_scores[:, row0_slot])
        card_scores[:, row1_slot] = torch.where(zero_mask, torch.zeros_like(card_scores[:, row1_slot]), card_scores[:, row1_slot])

    return card_scores.sum(dim=1)


def compute_final_score(cards: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute final score treating all cards as revealed."""
    N = cards.shape[0]
    all_revealed = torch.ones(N, 6, dtype=torch.bool, device=device)
    return compute_score(cards, all_revealed, device)


def get_observation(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Extract model tokens: [card0..5 or 52 if hidden, holding or 52, discard_top].

    Returns: (N, 8) int64
    """
    N = state.player_cards.shape[0]
    device = state.player_cards.device

    cards = state.player_cards[:, player_id, :].long()  # (N, 6)
    revealed = state.player_revealed[:, player_id, :]    # (N, 6)

    # Hidden cards show as UNKNOWN_CARD (52)
    obs_cards = torch.where(revealed, cards, torch.full_like(cards, UNKNOWN_CARD))

    # Holding: -1 means no card -> show as 52
    holding = state.player_holding[:, player_id].long()  # (N,)
    obs_holding = torch.where(holding >= 0, holding, torch.full_like(holding, UNKNOWN_CARD))

    # Discard top
    obs_discard = state.discard_top.long()  # (N,)

    return torch.cat([obs_cards, obs_holding.unsqueeze(1), obs_discard.unsqueeze(1)], dim=1)  # (N, 8)


def step_stage0(state: VectorizedGolfState, actions: torch.Tensor, player_id: int) -> torch.Tensor:
    """Execute stage 0 actions. Action 0=take face card, 1=draw from deck.

    Args:
        state: game state (modified in-place)
        actions: (N,) int64, 0 or 1
        player_id: which player is acting

    Returns:
        (N,) float32 rewards (score change)
    """
    N = actions.shape[0]
    device = actions.device

    # Compute score before action
    score_before = compute_score(
        state.player_cards[:, player_id, :],
        state.player_revealed[:, player_id, :],
        device,
    )

    take_face = (actions == 0)  # take face card
    draw_deck = (actions == 1)  # draw from deck

    # Take face card: holding = discard_top
    # Draw from deck: holding = deck[deck_ptr], deck_ptr += 1
    deck_card = state.deck[torch.arange(N, device=device), state.deck_ptr.long().clamp(max=NUM_CARDS - 1)]

    new_holding = torch.where(take_face, state.discard_top, deck_card)

    # Only update for non-done games
    active = ~state.done
    mask = active

    # Update holding
    state.player_holding[:, player_id] = torch.where(
        mask, new_holding, state.player_holding[:, player_id]
    )

    # Advance deck pointer for draw actions
    state.deck_ptr = torch.where(
        mask & draw_deck,
        state.deck_ptr + 1,
        state.deck_ptr,
    )

    # Update stage to 1
    state.current_stage = torch.where(
        mask, torch.ones_like(state.current_stage), state.current_stage
    )

    # Reward is 0 for stage 0 (no cards placed yet)
    return torch.zeros(N, dtype=torch.float32, device=device)


def step_stage1(state: VectorizedGolfState, actions: torch.Tensor, player_id: int) -> torch.Tensor:
    """Execute stage 1 actions.

    Actions 2-7: place held card at position 0-5 (replace existing card)
    Actions 9-14: discard held card + flip card at position 0-5

    Does NOT update last_turn/done -- callers handle game-over logic.

    Args:
        state: game state (modified in-place)
        actions: (N,) int64 action IDs (2-7 or 9-14)
        player_id: which player is acting

    Returns:
        (N,) float32 rewards (score improvement, positive = better)
    """
    N = actions.shape[0]
    device = actions.device
    active = ~state.done

    # Compute score before action
    score_before = compute_score(
        state.player_cards[:, player_id, :],
        state.player_revealed[:, player_id, :],
        device,
    )

    # Decode action: place (2-7) vs discard+flip (9-14)
    is_place = (actions >= 2) & (actions <= 7)
    is_discard_flip = (actions >= 9) & (actions <= 14)
    pos = torch.where(is_place, actions - 2, actions - 9)  # position 0-5
    pos = pos.clamp(0, 5)

    held_card = state.player_holding[:, player_id]  # (N,)

    # Get existing card at position
    existing_card = state.player_cards[:, player_id, :].gather(
        1, pos.long().unsqueeze(1)
    ).squeeze(1)  # (N,)

    # Place: replace card at pos with held card
    cards = state.player_cards[:, player_id, :].clone()
    cards = torch.where(
        (active & is_place).unsqueeze(1).expand_as(cards),
        cards.scatter(1, pos.long().unsqueeze(1), held_card.unsqueeze(1)),
        cards,
    )
    state.player_cards[:, player_id, :] = cards

    # Reveal the position (both place and flip reveal it)
    revealed = state.player_revealed[:, player_id, :].clone()
    pos_one_hot = torch.zeros_like(revealed)
    pos_one_hot.scatter_(1, pos.long().unsqueeze(1), True)
    revealed = revealed | (active.unsqueeze(1) & pos_one_hot)
    state.player_revealed[:, player_id, :] = revealed

    # Update discard top: place discards existing card, flip discards held card
    state.discard_top = torch.where(
        active,
        torch.where(is_place, existing_card, held_card),
        state.discard_top,
    )

    # Clear holding
    state.player_holding[:, player_id] = torch.where(
        active, torch.full_like(held_card, -1), state.player_holding[:, player_id]
    )

    # Reset stage to 0
    state.current_stage = torch.where(
        active, torch.zeros_like(state.current_stage), state.current_stage
    )

    # Compute score after
    score_after = compute_score(
        state.player_cards[:, player_id, :],
        state.player_revealed[:, player_id, :],
        device,
    )

    reward = score_before - score_after  # positive = improvement
    return torch.where(active, reward, torch.zeros_like(reward))


def get_valid_action_mask(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Get valid action mask for the current stage.

    Returns: (N, 16) bool
    """
    N = state.player_cards.shape[0]
    device = state.player_cards.device
    mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)

    stage = state.current_stage
    is_stage0 = (stage == 0)
    is_stage1 = (stage == 1)

    # Stage 0: action 0 (take face card) always valid, action 1 (draw) valid if deck not empty
    deck_not_empty = state.deck_ptr < NUM_CARDS
    mask[:, 0] = is_stage0  # take face card
    mask[:, 1] = is_stage0 & deck_not_empty  # draw from deck

    # Stage 1: actions 2-7 (place at pos 0-5) always valid
    for pos in range(6):
        mask[:, 2 + pos] = is_stage1  # place

    # Stage 1: actions 9-14 (discard+flip at pos 0-5) valid only if position is unrevealed
    revealed = state.player_revealed[:, player_id, :]  # (N, 6)
    for pos in range(6):
        mask[:, 9 + pos] = is_stage1 & (~revealed[:, pos])

    return mask


def heuristic_stage0(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Heuristic decision for stage 0: take face card or draw.

    Take face card if score < cutoff OR rank matches a revealed card.
    Else draw from deck.

    Returns: (N,) int64 action (0 or 1)
    """
    device = state.player_cards.device
    rank_scores = RANK_SCORES.to(device)
    N = state.player_cards.shape[0]

    face_rank = state.discard_top % NUM_RANKS  # (N,)
    face_score = rank_scores[face_rank.long()]  # (N,)

    # Check if face card score < cutoff
    take_low = face_score < RANK_CUTOFF  # (N,)

    # Check if face rank matches any revealed card rank
    player_cards = state.player_cards[:, player_id, :]  # (N, 6)
    player_revealed = state.player_revealed[:, player_id, :]  # (N, 6)
    player_ranks = player_cards % NUM_RANKS  # (N, 6)

    # Mask unrevealed cards
    revealed_ranks = torch.where(player_revealed, player_ranks, torch.full_like(player_ranks, -1))
    rank_match = (revealed_ranks == face_rank.unsqueeze(1)).any(dim=1)  # (N,)

    take_face = take_low | rank_match
    # 0 = take face card, 1 = draw from deck
    return torch.where(take_face, torch.zeros(N, dtype=torch.long, device=device),
                       torch.ones(N, dtype=torch.long, device=device))


def heuristic_stage1(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Heuristic decision for stage 1.

    Only tries UNREVEALED positions (matching iterative calc_opt_heuristic_position).
    Place if improvement >= cutoff, else place at first unrevealed if score
    doesn't get worse, else discard+flip first unrevealed.

    Returns: (N,) int64 action IDs (2-7 for place, 9-14 for discard+flip)
    """
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    cards = state.player_cards[:, player_id, :].clone()  # (N, 6)
    revealed = state.player_revealed[:, player_id, :].clone()  # (N, 6)
    held = state.player_holding[:, player_id]  # (N,)
    unrevealed = ~revealed  # (N, 6)

    current_score = compute_score(cards, revealed, device)  # (N,)

    # Try placing held card at each UNREVEALED position only
    best_score = torch.full((N,), 1e6, dtype=torch.float32, device=device)
    best_pos = torch.zeros(N, dtype=torch.long, device=device)

    for pos in range(6):
        is_unrevealed = unrevealed[:, pos]  # (N,)
        trial_cards = cards.clone()
        trial_cards[:, pos] = held
        trial_revealed = revealed.clone()
        trial_revealed[:, pos] = True
        score = compute_score(trial_cards, trial_revealed, device)  # (N,)
        # Only consider this position if it's unrevealed
        better = is_unrevealed & (score < best_score)
        best_score = torch.where(better, score, best_score)
        best_pos = torch.where(better, torch.full_like(best_pos, pos), best_pos)

    # Find first unrevealed position
    has_unrevealed = unrevealed.any(dim=1)
    unrevealed_idx = torch.where(
        unrevealed,
        torch.arange(6, device=device).unsqueeze(0).expand(N, -1),
        torch.full((N, 6), 99, dtype=torch.long, device=device),
    )
    first_unrevealed = unrevealed_idx.min(dim=1).values.clamp(0, 5)  # (N,)

    # Decision logic (matches get_player_action stage 1):
    # 1. If optimal_score <= current_score - rank_cutoff: place at best_pos
    big_improvement = best_score <= (current_score - RANK_CUTOFF)
    # 2. Elif (optimal_score - current_score) < rank_cutoff and has unrevealed: place at first unrevealed
    small_ok = (best_score - current_score) < RANK_CUTOFF
    place_unrevealed = (~big_improvement) & small_ok & has_unrevealed
    # 3. Else: discard + flip first unrevealed
    discard_flip = (~big_improvement) & (~place_unrevealed) & has_unrevealed

    # Default to place at best_pos
    action = 2 + best_pos

    # Override: place at first unrevealed
    action = torch.where(place_unrevealed, 2 + first_unrevealed, action)

    # Override: discard+flip first unrevealed
    action = torch.where(discard_flip, 9 + first_unrevealed, action)

    # Edge case: no unrevealed slots -> place at best_pos among revealed
    # (shouldn't normally happen with heuristic, but handle gracefully)
    if not has_unrevealed.all():
        # Fallback: try all positions for games with no unrevealed
        for pos in range(6):
            trial_cards = cards.clone()
            trial_cards[:, pos] = held
            score = compute_score(trial_cards, revealed, device)
            better = (~has_unrevealed) & (score < best_score)
            best_score = torch.where(better, score, best_score)
            best_pos = torch.where(better, torch.full_like(best_pos, pos), best_pos)
        no_unrevealed = ~has_unrevealed
        action = torch.where(no_unrevealed, 2 + best_pos, action)

    return action


def eps_greedy_batched(
    q_values: torch.Tensor,
    epsilon: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched epsilon-greedy with action masking.

    Args:
        q_values: (N, NUM_ACTIONS)
        epsilon: exploration rate
        valid_mask: (N, NUM_ACTIONS) bool

    Returns: (N,) int64 selected actions
    """
    N = q_values.shape[0]
    device = q_values.device

    # Greedy: pick best valid action
    masked_q = q_values.masked_fill(~valid_mask, float("-inf"))
    greedy_actions = masked_q.argmax(dim=1)

    # Random: pick random valid action
    # Use Gumbel trick for batched random selection from mask
    uniform = torch.rand(N, NUM_ACTIONS, device=device)
    uniform = uniform.masked_fill(~valid_mask, 0.0)
    random_actions = uniform.argmax(dim=1)

    # Epsilon-greedy selection
    explore = torch.rand(N, device=device) < epsilon
    return torch.where(explore, random_actions, greedy_actions)

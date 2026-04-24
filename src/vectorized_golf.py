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
    player_cards: torch.Tensor      # (N, P, 6) int16 -- card indices, P = n_players
    player_revealed: torch.Tensor   # (N, P, 6) bool -- which slots face-up
    player_holding: torch.Tensor    # (N, P) int16 -- held card, -1 if none
    deck: torch.Tensor              # (N, 52) int16 -- shuffled deck buffer (always 52 wide)
    deck_ptr: torch.Tensor          # (N,) int32 -- next card to draw
    discard_top: torch.Tensor       # (N,) int16 -- top of discard pile
    current_player: torch.Tensor    # (N,) int8 -- whose turn (0..P-1)
    current_stage: torch.Tensor     # (N,) int8 -- 0 or 1
    last_turn: torch.Tensor         # (N,) bool -- end-game triggered
    end_game_player: torch.Tensor   # (N,) int8 -- who triggered (-1 if not)
    done: torch.Tensor              # (N,) bool -- game finished
    scores: torch.Tensor            # (N, P) float32 -- cumulative scores
    # Number of valid cards in `deck` (cards at indices [deck_ptr, deck_size) are
    # drawable). After a discard-pile reshuffle this can be less than 52.
    deck_size: torch.Tensor = None  # (N,) int32 -- valid deck length
    # Cards in the discard pile UNDER the top (i.e., buried). When the deck
    # empties, these get reshuffled into a new deck. None for legacy paths
    # that never trigger reshuffle (4-player short games).
    discard_buried: torch.Tensor = None  # (N, 52) bool
    # Player count this state was created for. Stored as int (not tensor) so
    # callers can branch on it cheaply.
    n_players: int = 4


def reset_games(
    N: int,
    device: torch.device,
    n_players: int = 4,
    stack_low_cards: bool = False,
) -> VectorizedGolfState:
    """Create N fresh games: shuffle decks, deal 6 cards per player, set face card.

    Args:
        N: batch size.
        device: torch device.
        n_players: number of players (default 4). Each player gets 6 cards. The
            face card is dealt next, so n_players * 6 + 1 cards are committed
            from the deck up front. n_players is capped only by the deck:
            n_players * 6 + 1 <= 52, so up to 8 players from a single deck.
        stack_low_cards: if True, all low-score cards (rank 2, K, A; scores
            -2, 0, 1; 12 cards total) are moved to the END of the deck. As a
            result, the dealt cards and face card contain none of these
            low-score ranks, and the low cards only appear via deck draws
            after the high portion is exhausted. Used to construct rigged
            test scenarios where the bayes posterior should give a strong
            "low cards still in deck" signal late in the game.
    """
    if n_players * 6 + 1 > NUM_CARDS:
        raise ValueError(
            f"n_players={n_players} requires {n_players*6+1} cards but deck has {NUM_CARDS}"
        )

    # Generate shuffled decks: each row is a permutation of 0..51
    deck = torch.stack([torch.randperm(NUM_CARDS, device=device) for _ in range(N)])
    deck = deck.to(torch.int16)

    if stack_low_cards:
        # Move all rank-0 (2), rank-11 (K), rank-12 (A) cards to the end. Stable
        # sort by is_low (False before True) preserves the random within-group
        # order from randperm.
        ranks = (deck.long() % NUM_RANKS)
        is_low = (ranks == 0) | (ranks == 11) | (ranks == 12)
        sorted_idx = is_low.long().argsort(dim=1, stable=True)
        deck = deck.gather(1, sorted_idx)

    # Deal n_players * 6 cards: 6 per player.
    # Layout: player_cards[n, p, slot] where slot is 0..5 (row0: 0,1,2  row1: 3,4,5)
    n_dealt = n_players * 6
    deal_cards = deck[:, :n_dealt].reshape(N, n_players, 6)
    player_cards = deal_cards.clone()

    # All cards start face-down
    player_revealed = torch.zeros(N, n_players, 6, dtype=torch.bool, device=device)

    # No one is holding a card
    player_holding = torch.full((N, n_players), -1, dtype=torch.int16, device=device)

    # Face card is the next card after dealing
    discard_top = deck[:, n_dealt].clone()

    # Deck pointer starts after dealt cards + face card
    deck_ptr = torch.full((N,), n_dealt + 1, dtype=torch.int32, device=device)

    # Initial deck size = full deck (all 52 cards have a "location": dealt,
    # face card, or remaining in deck buffer).
    deck_size = torch.full((N,), NUM_CARDS, dtype=torch.int32, device=device)

    # Empty buried-pile mask -- only the face card is on top, nothing buried yet.
    discard_buried = torch.zeros(N, NUM_CARDS, dtype=torch.bool, device=device)

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
        scores=torch.zeros(N, n_players, dtype=torch.float32, device=device),
        deck_size=deck_size,
        discard_buried=discard_buried,
        n_players=n_players,
    )


def card_rank(card_indices: torch.Tensor) -> torch.Tensor:
    """Extract rank index (0-12) from card index (0-51). -1 for invalid."""
    return torch.where(card_indices >= 0, card_indices % NUM_RANKS, torch.tensor(-1, device=card_indices.device))


def riffle_shuffle(deck: torch.Tensor, n_riffles: int) -> torch.Tensor:
    """Apply n_riffles Gilbert-Shannon-Reeds riffle shuffles to a (N, 52) deck.

    Per riffle: cut the deck around the middle (with small jitter), then
    interleave by sampling from each half with probability proportional to
    the remaining cards in that half. This is the standard mathematical
    model of a casual physical riffle shuffle. Bayer-Diaconis (1992) showed
    that 7 riffles are sufficient to randomize a 52-card deck.

    Args:
        deck: (N, D) int16 tensor.
        n_riffles: number of riffle passes (>=0).

    Returns:
        new (N, D) int16 tensor with the shuffled deck.
    """
    if n_riffles <= 0:
        return deck.clone()

    N, D = deck.shape
    device = deck.device
    deck = deck.clone()

    for _ in range(n_riffles):
        # Cut around middle with small jitter (~ +/- D/16).
        jitter_range = max(1, D // 16)
        cut = torch.full((N,), D // 2, dtype=torch.long, device=device)
        cut += torch.randint(-jitter_range, jitter_range + 1, (N,), device=device)
        cut = cut.clamp(D // 4, 3 * D // 4)

        new_deck = torch.zeros_like(deck)
        left_ptr = torch.zeros(N, dtype=torch.long, device=device)
        right_ptr = cut.clone()
        rows = torch.arange(N, device=device)

        for i in range(D):
            left_remaining = (cut - left_ptr).clamp(min=0)
            right_remaining = (D - right_ptr).clamp(min=0)
            total_remaining = (left_remaining + right_remaining).clamp(min=1)
            left_prob = left_remaining.float() / total_remaining.float()
            rand = torch.rand(N, device=device)
            take_left = (rand < left_prob) & (left_remaining > 0)
            # If left exhausted, must take right; if right exhausted, must take left.
            take_left = torch.where(left_remaining == 0, torch.zeros_like(take_left), take_left)
            take_left = torch.where(right_remaining == 0, torch.ones_like(take_left), take_left)

            idx = torch.where(take_left, left_ptr, right_ptr)
            new_deck[:, i] = deck[rows, idx]

            left_ptr = torch.where(take_left, left_ptr + 1, left_ptr)
            right_ptr = torch.where(take_left, right_ptr, right_ptr + 1)

        deck = new_deck

    return deck


def reshuffle_empty_decks(state: VectorizedGolfState) -> torch.Tensor:
    """For games whose deck is empty, reshuffle the buried discard pile into
    a fresh deck. Updates state.deck, state.deck_ptr, state.deck_size, and
    state.discard_buried in place. Returns a (N,) bool mask of reshuffled games.

    The current discard top stays as the top -- only buried cards (everything
    under the top) get recycled.
    """
    if state.deck_size is None or state.discard_buried is None:
        return torch.zeros(state.player_cards.shape[0], dtype=torch.bool,
                           device=state.player_cards.device)

    N = state.player_cards.shape[0]
    device = state.player_cards.device

    empty = state.deck_ptr >= state.deck_size
    if not empty.any():
        return empty

    available = state.discard_buried  # (N, 52) bool

    # Randomize order of available cards via key sort. Non-available cards get
    # negative keys so they sort to the bottom; available cards land at the top.
    keys = torch.rand(N, NUM_CARDS, device=device)
    keys = torch.where(available, keys, torch.full_like(keys, -1.0))
    sorted_idx = keys.argsort(dim=1, descending=True)  # (N, 52)
    n_available = available.sum(dim=1).to(torch.int32)  # (N,)

    new_deck = torch.where(
        empty.unsqueeze(1),
        sorted_idx.to(torch.int16),
        state.deck,
    )
    state.deck = new_deck
    state.deck_ptr = torch.where(empty, torch.zeros_like(state.deck_ptr), state.deck_ptr)
    state.deck_size = torch.where(empty, n_available, state.deck_size)
    # Cards moved from buried pile into the deck buffer
    state.discard_buried = torch.where(
        empty.unsqueeze(1),
        torch.zeros_like(state.discard_buried),
        state.discard_buried,
    )

    return empty


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


def count_column_matches(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Count column matches (0-3) for a player based on revealed cards.

    A column match occurs when both cards in a column (slots i and i+3)
    are revealed and have the same rank.

    Returns: (N,) long tensor
    """
    ranks = state.player_cards[:, player_id, :] % NUM_RANKS
    revealed = state.player_revealed[:, player_id, :]
    N = state.player_cards.shape[0]
    matches = torch.zeros(N, dtype=torch.long, device=state.player_cards.device)
    for col in range(3):
        match = revealed[:, col] & revealed[:, col + 3] & (ranks[:, col] == ranks[:, col + 3])
        matches += match.long()
    return matches


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


def get_observation_v2(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Extract expanded observation with full table visibility.

    Layout (30 tokens):
        [0-5]   own 6 cards (revealed or 52)
        [6]     own holding (or 52)
        [7]     discard top
        [8-13]  opponent 1 visible cards (revealed or 52)
        [14]    opponent 1 holding (or 52)
        [15-20] opponent 2 visible cards (revealed or 52)
        [21]    opponent 2 holding (or 52)
        [22-27] opponent 3 visible cards (revealed or 52)
        [28]    opponent 3 holding (or 52)
        [29]    deck cards remaining (0-27, raw int)

    Opponents in relative order: (pid+1)%4, (pid+2)%4, (pid+3)%4.

    Returns: (N, 30) int64
    """
    N = state.player_cards.shape[0]
    device = state.player_cards.device

    # Own cards + holding + discard (same as v1)
    cards = state.player_cards[:, player_id, :].long()  # (N, 6)
    revealed = state.player_revealed[:, player_id, :]    # (N, 6)
    obs_cards = torch.where(revealed, cards, torch.full_like(cards, UNKNOWN_CARD))

    holding = state.player_holding[:, player_id].long()
    obs_holding = torch.where(holding >= 0, holding, torch.full_like(holding, UNKNOWN_CARD))

    obs_discard = state.discard_top.long()

    parts = [obs_cards, obs_holding.unsqueeze(1), obs_discard.unsqueeze(1)]

    # Opponent info in relative order
    for offset in range(1, 4):
        opp_id = (player_id + offset) % 4
        opp_cards = state.player_cards[:, opp_id, :].long()
        opp_revealed = state.player_revealed[:, opp_id, :]
        opp_obs = torch.where(opp_revealed, opp_cards, torch.full_like(opp_cards, UNKNOWN_CARD))

        opp_holding = state.player_holding[:, opp_id].long()
        opp_obs_holding = torch.where(opp_holding >= 0, opp_holding, torch.full_like(opp_holding, UNKNOWN_CARD))

        parts.append(opp_obs)
        parts.append(opp_obs_holding.unsqueeze(1))

    # Deck cards remaining
    deck_size = state.deck_size if state.deck_size is not None else torch.full_like(
        state.deck_ptr, NUM_CARDS
    )
    deck_remaining = (deck_size.long() - state.deck_ptr.long()).clamp(min=0)
    parts.append(deck_remaining.unsqueeze(1))

    return torch.cat(parts, dim=1)  # (N, 30)


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

    # If any drawing game has an empty deck, reshuffle the discard pile.
    # We only need to reshuffle for games that are about to draw AND are empty.
    if state.deck_size is not None:
        needs_reshuffle = draw_deck & (state.deck_ptr >= state.deck_size) & (~state.done)
        if needs_reshuffle.any():
            reshuffle_empty_decks(state)

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

    # If the player drew from the deck (held_card != current discard_top),
    # the current discard_top is about to be covered -- mark it as buried so
    # the reshuffle can recover it later. If they took the face card,
    # held_card == discard_top and the card is moving from the top into
    # their slot (place) or staying on top (flip+discard back), neither of
    # which buries the original top.
    if state.discard_buried is not None:
        old_top = state.discard_top.long().clamp(0, NUM_CARDS - 1)
        drew_from_deck = (held_card != state.discard_top) & active
        rows = torch.arange(N, device=device)
        existing_buried = state.discard_buried[rows, old_top]
        state.discard_buried[rows, old_top] = existing_buried | drew_from_deck

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

    # Stage 0: action 0 (take face card) always valid. Action 1 (draw) valid
    # if the deck has cards. With deck reshuffling enabled (deck_size set),
    # we treat draw as always valid -- reshuffle_empty_decks (called inside
    # step_stage0) refills the deck from the discard pile if empty. The
    # only way it would still be empty after reshuffle is if every card is
    # face-up in someone's layout, which means the game is essentially over.
    if state.deck_size is not None:
        deck_not_empty = torch.ones(N, dtype=torch.bool, device=device)
    else:
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


def heuristic_stage0(state: VectorizedGolfState, player_id: int, cutoff: float = RANK_CUTOFF) -> torch.Tensor:
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
    take_low = face_score < cutoff  # (N,)

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


def heuristic_stage1(state: VectorizedGolfState, player_id: int, cutoff: float = RANK_CUTOFF) -> torch.Tensor:
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
    # 1. If optimal_score <= current_score - cutoff: place at best_pos
    big_improvement = best_score <= (current_score - cutoff)
    # 2. Elif (optimal_score - current_score) < cutoff and has unrevealed: place at first unrevealed
    small_ok = (best_score - current_score) < cutoff
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


# ---------------------------------------------------------------------------
# Variant heuristics
# ---------------------------------------------------------------------------


def simple_stage0(state: VectorizedGolfState, player_id: int, cutoff: float = RANK_CUTOFF) -> torch.Tensor:
    """Take face card if score < cutoff, else draw. No rank-matching."""
    device = state.player_cards.device
    rank_scores = RANK_SCORES.to(device)
    N = state.player_cards.shape[0]

    face_rank = state.discard_top % NUM_RANKS
    face_score = rank_scores[face_rank.long()]
    take = face_score < cutoff
    return torch.where(take, torch.zeros(N, dtype=torch.long, device=device),
                       torch.ones(N, dtype=torch.long, device=device))


def simple_stage1(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Place held card at a random unrevealed position, or random position if all revealed."""
    device = state.player_cards.device
    N = state.player_cards.shape[0]
    revealed = state.player_revealed[:, player_id, :]  # (N, 6)
    unrevealed = ~revealed

    # Random priority per position, masked by unrevealed
    rand = torch.rand(N, 6, device=device)
    rand = torch.where(unrevealed, rand, torch.full_like(rand, -1.0))

    has_unrevealed = unrevealed.any(dim=1)
    # If all revealed, allow any position
    rand = torch.where(has_unrevealed.unsqueeze(1), rand, torch.rand(N, 6, device=device))

    pos = rand.argmax(dim=1)  # (N,)
    return 2 + pos  # always place (action 2-7)


def improved_stage1(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Like heuristic_stage1 but considers ALL positions (including revealed)."""
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    cards = state.player_cards[:, player_id, :].clone()
    revealed = state.player_revealed[:, player_id, :].clone()
    held = state.player_holding[:, player_id]
    unrevealed = ~revealed

    current_score = compute_score(cards, revealed, device)

    # Try placing at ALL 6 positions
    best_score = torch.full((N,), 1e6, dtype=torch.float32, device=device)
    best_pos = torch.zeros(N, dtype=torch.long, device=device)

    for pos in range(6):
        trial_cards = cards.clone()
        trial_cards[:, pos] = held
        trial_revealed = revealed.clone()
        trial_revealed[:, pos] = True
        score = compute_score(trial_cards, trial_revealed, device)
        better = score < best_score
        best_score = torch.where(better, score, best_score)
        best_pos = torch.where(better, torch.full_like(best_pos, pos), best_pos)

    # Find first unrevealed position (for discard+flip fallback)
    has_unrevealed = unrevealed.any(dim=1)
    unrevealed_idx = torch.where(
        unrevealed,
        torch.arange(6, device=device).unsqueeze(0).expand(N, -1),
        torch.full((N, 6), 99, dtype=torch.long, device=device),
    )
    first_unrevealed = unrevealed_idx.min(dim=1).values.clamp(0, 5)

    # 1. Big improvement: place at best_pos
    big_improvement = best_score <= (current_score - RANK_CUTOFF)
    # 2. Small/neutral: place at first unrevealed for info gain
    small_ok = (best_score - current_score) < RANK_CUTOFF
    place_unrevealed = (~big_improvement) & small_ok & has_unrevealed
    # 3. Else: discard + flip
    discard_flip = (~big_improvement) & (~place_unrevealed) & has_unrevealed

    action = 2 + best_pos
    action = torch.where(place_unrevealed, 2 + first_unrevealed, action)
    action = torch.where(discard_flip, 9 + first_unrevealed, action)

    # No unrevealed: place at best_pos (already the default)
    return action


def random_stage0(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Random stage 0: uniformly pick a valid action (take face or draw)."""
    N = state.player_cards.shape[0]
    device = state.player_cards.device
    deck_not_empty = state.deck_ptr < NUM_CARDS
    # If deck empty, must take face (action 0); else uniform over {0, 1}
    draw = deck_not_empty & (torch.rand(N, device=device) < 0.5)
    return draw.long()


def random_stage1(state: VectorizedGolfState, player_id: int) -> torch.Tensor:
    """Random stage 1: uniformly pick a valid action (place or discard+flip)."""
    N = state.player_cards.shape[0]
    device = state.player_cards.device
    revealed = state.player_revealed[:, player_id, :]  # (N, 6)

    # Build valid action mask: 2-7 always valid, 9-14 valid if unrevealed
    mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)
    for pos in range(6):
        mask[:, 2 + pos] = True
        mask[:, 9 + pos] = ~revealed[:, pos]

    uniform = torch.rand(N, NUM_ACTIONS, device=device)
    uniform.masked_fill_(~mask, 0.0)
    return uniform.argmax(dim=1)


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

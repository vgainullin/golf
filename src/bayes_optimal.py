"""Belief-augmented heuristic Golf player.

Takes the improved heuristic (`improved_stage1` + `heuristic_stage0`) and
replaces its hard-coded constants with values derived from a posterior over
unobserved cards. The belief is exact: under shuffle-once-and-deal, every
unobserved card has the same uniform-over-unobserved-multiset posterior, so a
single (N, 52) bool mask is sufficient.

Two changes vs the improved heuristic:

1. Stage 0: cutoff for "take face card" becomes E[score(unknown)] under the
   current belief, instead of the constant RANK_CUTOFF=4. The rank-match rule
   is preserved.

2. Stage 1: trial layouts are scored with `expected_score`, which treats each
   face-down slot as a draw from the belief multiset (rather than zero, as
   `compute_score` does). The same big-improvement / small-ok / discard-flip
   decision rules are used, but on a more accurate score signal.

The belief tracker observes every visible card across all four players on
every turn so that ephemeral discards (briefly on top, then taken by an
opponent) are still recorded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from src.vectorized_golf import (
    NUM_CARDS,
    NUM_RANKS,
    RANK_CUTOFF,
    RANK_SCORES,
    VectorizedGolfState,
    compute_final_score,
    compute_score,
    heuristic_stage0,
    heuristic_stage1,
    improved_stage1,
    random_stage0,
    random_stage1,
    reset_games,
    step_stage0,
    step_stage1,
)


# ---------------------------------------------------------------------------
# Belief tracker
# ---------------------------------------------------------------------------


class BayesBeliefTracker:
    """Tracks the set of still-unobserved cards per game in a batch.

    The belief is a (N, 52) bool mask: True = the card has not yet been seen
    by us. From this we derive a per-rank multiset and an expected score for
    any unknown card.
    """

    def __init__(self, N: int, device: torch.device):
        self.N = N
        self.device = device
        # All 52 cards start unobserved
        self.unobserved = torch.ones(N, NUM_CARDS, dtype=torch.bool, device=device)
        # Precompute rank-of-card lookup
        self._card_ranks = (torch.arange(NUM_CARDS, device=device) % NUM_RANKS).long()
        self._rank_scores = RANK_SCORES.to(device)

    def reset(self) -> None:
        """Mark all 52 cards unobserved (call at the start of each hole)."""
        self.unobserved.fill_(True)

    def observe(self, state: VectorizedGolfState, my_player_id: int) -> None:
        """Remove from `unobserved` every card visible to player `my_player_id`.

        Visible cards from this player's POV:
          - face-up cards in any player's layout (including own),
          - the current discard top,
          - my own holding (if any). Other players' holdings are NOT visible
            to me -- they're either unknown deck draws or cards already on the
            discard pile.

        Idempotent. Call after every step (own and opponents') so cards that
        spend a brief moment on the discard pile are still captured.
        """
        N = self.N
        device = self.device
        n_players = state.player_cards.shape[1]

        all_cards = state.player_cards.reshape(N, -1).long()  # (N, n_players*6)
        all_revealed = state.player_revealed.reshape(N, -1)

        seen = torch.zeros(N, NUM_CARDS, dtype=torch.bool, device=device)
        row = torch.arange(N, device=device)

        for slot in range(all_cards.shape[1]):
            card_idx = all_cards[:, slot].clamp(0, NUM_CARDS - 1)
            slot_revealed = all_revealed[:, slot]
            seen[row, card_idx] |= slot_revealed

        # Discard top is face-up to everyone.
        discard_idx = state.discard_top.long().clamp(0, NUM_CARDS - 1)
        seen[row, discard_idx] = True

        # Own holding only (we don't see opponents' held cards).
        my_holding = state.player_holding[:, my_player_id].long()
        my_valid = my_holding >= 0
        my_holding_clamped = my_holding.clamp(0, NUM_CARDS - 1)
        seen[row, my_holding_clamped] |= my_valid

        self.unobserved &= ~seen

    # ----- derived quantities -----

    def multiset_by_rank(self) -> torch.Tensor:
        """(N, 13) int counts of unobserved cards per rank."""
        # unobserved: (N, 52). Reshape via rank lookup: counts[n, r] = sum_{c: rank(c)=r} unobserved[n, c].
        # Use a one-hot rank matrix (52, 13) and matmul.
        if not hasattr(self, "_rank_one_hot"):
            self._rank_one_hot = torch.nn.functional.one_hot(
                self._card_ranks, num_classes=NUM_RANKS
            ).float()  # (52, 13)
        return (self.unobserved.float() @ self._rank_one_hot).to(torch.int64)

    def total(self) -> torch.Tensor:
        """(N,) int total unobserved cards per game."""
        return self.unobserved.sum(dim=1)

    def expected_unknown_score(self) -> torch.Tensor:
        """(N,) E[rank score | uniform over unobserved cards]. Returns 0 where total=0."""
        per_card_score = self._rank_scores[self._card_ranks]  # (52,)
        score_sum = (self.unobserved.float() @ per_card_score)  # (N,)
        total = self.total().clamp(min=1).float()
        return score_sum / total


# ---------------------------------------------------------------------------
# Belief-aware scoring
# ---------------------------------------------------------------------------


def expected_score(
    cards: torch.Tensor,
    revealed: torch.Tensor,
    multiset: torch.Tensor,
    total: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Expected final score of a 6-card layout under the belief multiset.

    For each of the 3 columns (slot c paired with slot c+3) we have:
      - both revealed: deterministic (0 if column-match, else sum of rank scores).
      - one revealed (rank r), one hidden ?:
            E = P(?=r) * 0 + (1 - P(?=r)) * (score(r) + E[score(?) | ? != r])
      - both hidden: E[s(a)+s(b)] - 2 * sum_r P(both=r) * score(r), with both
        cards sampled WITHOUT replacement from the unobserved multiset.

    The cross-column dependence (different columns share the same multiset)
    is ignored, which is a small approximation when |multiset| >> 6.

    Args:
        cards: (N, 6) int card indices
        revealed: (N, 6) bool
        multiset: (N, 13) int counts per rank of unobserved cards
        total: (N,) int total unobserved
        device: torch device

    Returns:
        (N,) float32 expected final score
    """
    N = cards.shape[0]
    rank_scores = RANK_SCORES.to(device)  # (13,)
    ranks = (cards % NUM_RANKS).long()  # (N, 6), valid only where revealed

    multiset_f = multiset.float()  # (N, 13)
    total_f = total.float().clamp(min=1)  # (N,)
    total_minus1 = (total.float() - 1).clamp(min=1)  # (N,)

    # E[score(?)]                           = sum_r (n_r/N) * score(r)
    e_unknown = (multiset_f * rank_scores.unsqueeze(0)).sum(dim=1) / total_f  # (N,)

    # P(both unknowns share rank r) under sample-w/o-replacement:
    #   n_r * (n_r - 1) / (N * (N - 1))
    p_pair_per_rank = (multiset_f * (multiset_f - 1)) / (
        (total_f * total_minus1).unsqueeze(1)
    )
    # P(both unknowns share rank r), shape (N, 13)
    e_both_hidden_correction = (p_pair_per_rank * rank_scores.unsqueeze(0)).sum(dim=1)  # (N,)

    # E[col | both hidden] = 2 * E[?] - 2 * correction
    e_col_both_hidden = 2.0 * e_unknown - 2.0 * e_both_hidden_correction  # (N,)

    out = torch.zeros(N, dtype=torch.float32, device=device)

    for col in range(3):
        a_idx = col
        b_idx = col + 3
        a_rev = revealed[:, a_idx]  # (N,)
        b_rev = revealed[:, b_idx]
        a_rank = ranks[:, a_idx]
        b_rank = ranks[:, b_idx]

        a_score = rank_scores[a_rank]
        b_score = rank_scores[b_rank]

        # ----- both revealed -----
        both_rev = a_rev & b_rev
        match = a_rank == b_rank
        col_both_rev = torch.where(
            match, torch.zeros_like(a_score), a_score + b_score
        )

        # ----- a revealed, b hidden -----
        # P(? = a_rank) = n_{a_rank} / N
        n_at_a_rank = multiset_f.gather(1, a_rank.clamp(min=0).unsqueeze(1)).squeeze(1)  # (N,)
        p_match_a = n_at_a_rank / total_f  # (N,)
        # E[score(?) | ? != a_rank] = (sum_r n_r*score(r) - n_{a_rank}*score(a_rank)) / (N - n_{a_rank})
        denom_a = (total_f - n_at_a_rank).clamp(min=1)
        sum_n_score = (multiset_f * rank_scores.unsqueeze(0)).sum(dim=1)  # (N,)
        e_other_a = (sum_n_score - n_at_a_rank * a_score) / denom_a
        col_a_rev_only = (1.0 - p_match_a) * (a_score + e_other_a)  # match=>0

        # ----- b revealed, a hidden -----
        n_at_b_rank = multiset_f.gather(1, b_rank.clamp(min=0).unsqueeze(1)).squeeze(1)
        p_match_b = n_at_b_rank / total_f
        denom_b = (total_f - n_at_b_rank).clamp(min=1)
        e_other_b = (sum_n_score - n_at_b_rank * b_score) / denom_b
        col_b_rev_only = (1.0 - p_match_b) * (b_score + e_other_b)

        # ----- both hidden -----
        # already computed above as e_col_both_hidden

        a_only = a_rev & (~b_rev)
        b_only = b_rev & (~a_rev)
        none = (~a_rev) & (~b_rev)

        col_score = torch.zeros_like(out)
        col_score = torch.where(both_rev, col_both_rev, col_score)
        col_score = torch.where(a_only, col_a_rev_only, col_score)
        col_score = torch.where(b_only, col_b_rev_only, col_score)
        col_score = torch.where(none, e_col_both_hidden, col_score)

        out = out + col_score

    return out


# ---------------------------------------------------------------------------
# Belief-aware stage functions
# ---------------------------------------------------------------------------


def bayes_stage0(
    state: VectorizedGolfState,
    player_id: int,
    tracker: BayesBeliefTracker,
    use_belief: bool = True,
) -> torch.Tensor:
    """Belief-aware stage 0.

    Take face card if its rank score < E[score of an unknown card] OR if its
    rank matches a revealed card in our layout. Otherwise draw from deck.

    The "<" cutoff is the only difference from `heuristic_stage0`: a constant
    4 is replaced by the belief-derived expected unknown score.

    If use_belief=False, the cutoff falls back to the constant RANK_CUTOFF=4,
    making this function equivalent to `heuristic_stage0`. This is the
    "ablation" path: with belief disabled, the bayes player must produce
    identical decisions to the improved heuristic.
    """
    device = state.player_cards.device
    rank_scores = RANK_SCORES.to(device)
    N = state.player_cards.shape[0]

    face_rank = (state.discard_top % NUM_RANKS).long()  # (N,)
    face_score = rank_scores[face_rank]  # (N,)

    if use_belief:
        e_unknown = tracker.expected_unknown_score()  # (N,)
    else:
        e_unknown = torch.full((N,), float(RANK_CUTOFF), device=device)

    take_low = face_score < e_unknown

    player_cards = state.player_cards[:, player_id, :]
    player_revealed = state.player_revealed[:, player_id, :]
    player_ranks = player_cards % NUM_RANKS
    revealed_ranks = torch.where(
        player_revealed, player_ranks, torch.full_like(player_ranks, -1)
    )
    rank_match = (revealed_ranks == face_rank.unsqueeze(1)).any(dim=1)

    take_face = take_low | rank_match
    return torch.where(
        take_face,
        torch.zeros(N, dtype=torch.long, device=device),
        torch.ones(N, dtype=torch.long, device=device),
    )


def bayes_v2_stage0(
    state: VectorizedGolfState,
    player_id: int,
    tracker: BayesBeliefTracker,
    cutoff: float = float(RANK_CUTOFF),
) -> torch.Tensor:
    """Stage 0 = improved heuristic + a belief-aware take rule.

    This is a strict superset of `heuristic_stage0`. It evaluates the
    EXPECTED net cost of taking the face card, accounting for the
    probability that placing it creates a column match:

        expected_face_cost = face_score * (1 - 2 * p_per_card)
        take_face if expected_face_cost < cutoff

    where:
      p_per_card = multiset[face_rank] / total (the per-slot posterior that
                   any specific hidden own slot has the face card's rank).

    Reasoning: if we take the face card and place it on slot A in a column
    where slot B is hidden, the column matches iff slot B == face_rank.
    P(match) = P(slot B = face_rank) = multiset[face_rank] / total.
    Savings from a match: both slots score 0 instead of 2 * face_score
    (the column was going to score face_score from our placement plus
    face_score from the matched hidden card). So savings = 2 * face_score.
    Expected net cost of taking = face_score - p * 2 * face_score
                                = face_score * (1 - 2p).

    Comparison threshold: the constant `cutoff` (default 4) is the same
    threshold IH uses, calibrated against the EV of drawing (which includes
    the optionality of flip+discarding a bad draw).

    With p_per_card = 0 the rule reduces to IH's `face_score < cutoff`,
    plus the IH revealed-rank-match clause. Strict superset of IH.
    """
    device = state.player_cards.device
    rank_scores = RANK_SCORES.to(device)
    N = state.player_cards.shape[0]

    face_rank = (state.discard_top % NUM_RANKS).long()  # (N,)
    face_score = rank_scores[face_rank]  # (N,)

    # IH rule: rank matches a revealed own card. (Deterministic match opportunity.)
    player_cards = state.player_cards[:, player_id, :]
    player_revealed = state.player_revealed[:, player_id, :]
    player_ranks = player_cards % NUM_RANKS
    revealed_ranks = torch.where(
        player_revealed, player_ranks, torch.full_like(player_ranks, -1)
    )
    rank_match_revealed = (revealed_ranks == face_rank.unsqueeze(1)).any(dim=1)

    # Belief-aware EV check.
    # p_per_card = multiset[face_rank] / total: marginal probability that any
    # specific hidden slot has the face card's rank. Bounded by [0, 1].
    multiset = tracker.multiset_by_rank().float()  # (N, 13)
    total = tracker.total().float().clamp(min=1)  # (N,)
    n_at_face = multiset.gather(1, face_rank.unsqueeze(1)).squeeze(1)  # (N,)
    p_per_card = n_at_face / total  # (N,)

    # Only meaningful if we have a hidden slot to place into. If all 6 slots
    # are face-up there is no column-match-via-placement opportunity.
    has_hidden = (~player_revealed).any(dim=1)
    p_per_card = torch.where(has_hidden, p_per_card, torch.zeros_like(p_per_card))

    expected_face_cost = face_score * (1.0 - 2.0 * p_per_card)
    take_belief = expected_face_cost < cutoff

    take_face = take_belief | rank_match_revealed
    return torch.where(
        take_face,
        torch.zeros(N, dtype=torch.long, device=device),
        torch.ones(N, dtype=torch.long, device=device),
    )


def bayes_v3_stage0(
    state: VectorizedGolfState,
    player_id: int,
    tracker: BayesBeliefTracker,
    draw_override_threshold: float = 0.50,
) -> torch.Tensor:
    """Stage 0 = improved heuristic with a belief-driven DRAW OVERRIDE.

    NOT a strict superset of IH. Starts from IH's defaults but OVERRIDES
    IH's "take low face card" with a draw when the posterior says the
    deck is very likely to give a strictly better card:

        if P(deck_draw_score < face_score | belief) > threshold
        AND IH would have taken purely for being low (not for a deterministic
            revealed-rank match)
        then DRAW instead

    The deterministic rank-match path (face_rank present in own face-up
    layout) is NOT overridden -- a deterministic column match is almost
    always better than the gamble of drawing.

    Reasoning: in regimes where the deck is enriched in low-score cards
    (e.g., the stacked-deck test, or simply late game when high cards have
    been disproportionately observed), a low face card on the discard top
    may still be worse than a typical draw. The IH cutoff doesn't see
    this; the belief does.

    With draw_override_threshold = 1.01 the override never fires and the
    function reduces to `heuristic_stage0` exactly.
    """
    device = state.player_cards.device
    rank_scores = RANK_SCORES.to(device)
    N = state.player_cards.shape[0]

    face_rank = (state.discard_top % NUM_RANKS).long()  # (N,)
    face_score = rank_scores[face_rank]  # (N,)

    # IH rule 1: low card.
    take_low = face_score < RANK_CUTOFF

    # IH rule 2: rank matches a revealed own card (deterministic match).
    player_cards = state.player_cards[:, player_id, :]
    player_revealed = state.player_revealed[:, player_id, :]
    player_ranks = player_cards % NUM_RANKS
    revealed_ranks = torch.where(
        player_revealed, player_ranks, torch.full_like(player_ranks, -1)
    )
    rank_match_revealed = (revealed_ranks == face_rank.unsqueeze(1)).any(dim=1)

    # P(deck draw is strictly better than face card | belief).
    multiset = tracker.multiset_by_rank().float()  # (N, 13)
    total = tracker.total().float().clamp(min=1)  # (N,)
    lt_mask = rank_scores.unsqueeze(0) < face_score.unsqueeze(1)  # (N, 13)
    p_draw_lt = (multiset * lt_mask.float()).sum(dim=1) / total  # (N,)

    high_p_lt = p_draw_lt > draw_override_threshold

    # Take if rank_match (deterministic), OR if take_low AND not overridden.
    take_face = rank_match_revealed | (take_low & ~high_p_lt)
    return torch.where(
        take_face,
        torch.zeros(N, dtype=torch.long, device=device),
        torch.ones(N, dtype=torch.long, device=device),
    )


def bayes_stage1(
    state: VectorizedGolfState,
    player_id: int,
    tracker: BayesBeliefTracker,
    use_belief: bool = True,
) -> torch.Tensor:
    """Belief-aware stage 1.

    Same structure as `improved_stage1`, but trial layouts are scored with
    `expected_score` (which treats face-down slots as belief-multiset draws)
    rather than `compute_score` (which treats them as zero).

    For each candidate placement position p, simulate placing the held card
    at p (revealing slot p, removing the displaced card from the multiset)
    and compute E[final score]. Pick the position that minimizes this.

    Decision rules carry over:
      1. If best E[final] <= current E[final] - cutoff: place at best_pos.
      2. Elif best E[final] - current E[final] < cutoff and we have an
         unrevealed slot: place at the first unrevealed slot for info gain.
      3. Else: discard + flip the first unrevealed slot.
    """
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    cards = state.player_cards[:, player_id, :].clone()  # (N, 6)
    revealed = state.player_revealed[:, player_id, :].clone()  # (N, 6)
    held = state.player_holding[:, player_id]  # (N,)
    unrevealed = ~revealed

    multiset = tracker.multiset_by_rank()  # (N, 13)
    total = tracker.total()  # (N,)

    def score_fn(c, r):
        if use_belief:
            return expected_score(c, r, multiset, total, device)
        else:
            # Belief-disabled: use compute_score (face-down slots = 0).
            # This makes bayes_stage1 byte-for-byte equivalent to improved_stage1.
            return compute_score(c, r, device)

    # Current expected final score under current belief (no action taken).
    current_e = score_fn(cards, revealed)  # (N,)

    # For each candidate position, the held card is placed (becomes revealed).
    # The displaced card at that position leaves the deck/our view; if the
    # position was previously unrevealed, the displaced card is one we never
    # saw, so the multiset effectively shrinks by one *random* card. We
    # approximate this as removing one expected-rank card -- which we model by
    # passing the same multiset (since removing one uniform sample doesn't
    # change the per-rank proportions in expectation). For the small bias
    # this introduces, we accept it as a baseline-quality approximation.
    #
    # If the position was already revealed, the displaced card is known and
    # has been observed before; the multiset is unchanged.
    #
    # The held card itself is already in `unobserved=False` because the
    # tracker observed it when we picked it up. So no further multiset
    # update is needed for the held card.

    best_score = torch.full((N,), 1e6, dtype=torch.float32, device=device)
    best_pos = torch.zeros(N, dtype=torch.long, device=device)

    for pos in range(6):
        trial_cards = cards.clone()
        trial_cards[:, pos] = held
        trial_revealed = revealed.clone()
        trial_revealed[:, pos] = True
        score = score_fn(trial_cards, trial_revealed)
        better = score < best_score
        best_score = torch.where(better, score, best_score)
        best_pos = torch.where(better, torch.full_like(best_pos, pos), best_pos)

    # First unrevealed position (for fallback flip)
    has_unrevealed = unrevealed.any(dim=1)
    unrevealed_idx = torch.where(
        unrevealed,
        torch.arange(6, device=device).unsqueeze(0).expand(N, -1),
        torch.full((N, 6), 99, dtype=torch.long, device=device),
    )
    first_unrevealed = unrevealed_idx.min(dim=1).values.clamp(0, 5)

    big_improvement = best_score <= (current_e - RANK_CUTOFF)
    small_ok = (best_score - current_e) < RANK_CUTOFF
    place_unrevealed = (~big_improvement) & small_ok & has_unrevealed
    discard_flip = (~big_improvement) & (~place_unrevealed) & has_unrevealed

    action = 2 + best_pos
    action = torch.where(place_unrevealed, 2 + first_unrevealed, action)
    action = torch.where(discard_flip, 9 + first_unrevealed, action)

    return action


# ---------------------------------------------------------------------------
# 1-step lookahead (threshold-free)
# ---------------------------------------------------------------------------


def _best_placement_score(
    cards: torch.Tensor,
    revealed: torch.Tensor,
    held: torch.Tensor,
    multiset: torch.Tensor,
    total: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the placement position minimizing expected final score.

    Returns (best_score, best_pos) each (N,).
    """
    N = cards.shape[0]
    best_score = torch.full((N,), 1e6, dtype=torch.float32, device=device)
    best_pos = torch.zeros(N, dtype=torch.long, device=device)

    for pos in range(6):
        trial_cards = cards.clone()
        trial_cards[:, pos] = held
        trial_revealed = revealed.clone()
        trial_revealed[:, pos] = True
        score = expected_score(trial_cards, trial_revealed, multiset, total, device)
        better = score < best_score
        best_score = torch.where(better, score, best_score)
        best_pos = torch.where(better, torch.full_like(best_pos, pos), best_pos)

    return best_score, best_pos


def lookahead_stage1(
    state: VectorizedGolfState,
    player_id: int,
    tracker: BayesBeliefTracker,
) -> torch.Tensor:
    """Pick the stage-1 action minimizing expected final score.

    Enumerates all 6 placement positions and compares against the
    discard+flip alternative. No thresholds or cutoffs.

    Discard+flip doesn't change E[final score] (by iterated expectations:
    revealing a hidden card reduces variance but not the mean), so its
    expected score equals the current layout's expected score. The player
    should place when placement improves on that, otherwise discard+flip.
    """
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    cards = state.player_cards[:, player_id, :].clone()
    revealed = state.player_revealed[:, player_id, :].clone()
    held = state.player_holding[:, player_id]

    multiset = tracker.multiset_by_rank()
    total = tracker.total()

    current_e = expected_score(cards, revealed, multiset, total, device)
    best_place_score, best_place_pos = _best_placement_score(
        cards, revealed, held, multiset, total, device
    )

    has_unrevealed = (~revealed).any(dim=1)
    unrevealed_idx = torch.where(
        ~revealed,
        torch.arange(6, device=device).unsqueeze(0).expand(N, -1),
        torch.full((N, 6), 99, dtype=torch.long, device=device),
    )
    first_unrevealed = unrevealed_idx.min(dim=1).values.clamp(0, 5)

    place_is_better = best_place_score < current_e

    # Place at best pos if it improves score, otherwise discard+flip.
    # If no unrevealed slots exist, must place regardless.
    action = torch.where(
        place_is_better | (~has_unrevealed),
        2 + best_place_pos,
        9 + first_unrevealed,
    )
    return action


def lookahead_stage0(
    state: VectorizedGolfState,
    player_id: int,
    tracker: BayesBeliefTracker,
) -> torch.Tensor:
    """Pick take (0) or draw (1) by comparing expected final scores.

    Take branch: face card is known; simulate optimal placement.
    Draw branch: average over all 13 possible ranks weighted by belief,
    with multiset adjusted for the drawn card.
    """
    device = state.player_cards.device
    N = state.player_cards.shape[0]
    rank_scores_t = RANK_SCORES.to(device)

    cards = state.player_cards[:, player_id, :].clone()
    revealed = state.player_revealed[:, player_id, :].clone()

    multiset = tracker.multiset_by_rank()  # (N, 13)
    total = tracker.total()  # (N,)
    total_f = total.float().clamp(min=1)

    # Current expected score (baseline for discard+flip).
    current_e = expected_score(cards, revealed, multiset, total, device)

    # --- Take branch ---
    face_card = state.discard_top.long()
    best_take_score, _ = _best_placement_score(
        cards, revealed, face_card, multiset, total, device
    )
    e_take = torch.min(best_take_score, current_e)

    # --- Draw branch ---
    # For each rank r, compute the best outcome (place or discard+flip)
    # weighted by P(drawing rank r).
    e_draw = torch.zeros(N, dtype=torch.float32, device=device)

    for r in range(NUM_RANKS):
        count_r = multiset[:, r].float()  # (N,)
        p_r = count_r / total_f  # (N,)

        # Skip rank entirely if no game has it in the unobserved set.
        if (count_r == 0).all():
            continue

        # Adjusted multiset after drawing rank r.
        draw_ms = multiset.clone()
        draw_ms[:, r] = (draw_ms[:, r] - 1).clamp(min=0)
        draw_total = (total - 1).clamp(min=1)

        # Virtual card with rank r (suit 0 -> card index = r).
        virtual_held = torch.full((N,), r, dtype=torch.long, device=device)

        best_draw_score, _ = _best_placement_score(
            cards, revealed, virtual_held, draw_ms, draw_total, device
        )

        # Discard+flip baseline with adjusted multiset.
        current_e_r = expected_score(cards, revealed, draw_ms, draw_total, device)
        e_draw_r = torch.min(best_draw_score, current_e_r)

        e_draw = e_draw + p_r * e_draw_r

    return torch.where(
        e_take <= e_draw,
        torch.zeros(N, dtype=torch.long, device=device),
        torch.ones(N, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


SeatFn = Tuple[Callable, Callable]


SEAT_PRESETS = {
    "R": (random_stage0, random_stage1),
    "H": (heuristic_stage0, improved_stage1),  # "H" = improved heuristic
    "h": (heuristic_stage0, heuristic_stage1),  # base heuristic
}


def parse_eval_config(spec: str) -> List[str]:
    """Parse 'R,H,R' into ['R', 'H', 'R']. Seat 0 is always Bayes; the parsed
    list is the opponents (length n_players - 1)."""
    parts = [p.strip() for p in spec.split(",")]
    if not parts:
        raise ValueError("--eval-config must contain at least one opponent")
    for p in parts:
        if p not in SEAT_PRESETS:
            raise ValueError(f"Unknown seat preset {p!r}; valid: {list(SEAT_PRESETS)}")
    return parts


def run_bayes_eval(
    opponent_specs: List[str],
    num_games: int,
    holes: int,
    device: torch.device,
    n_players: int = 4,
    player: str = "bayes",
) -> float:
    """Run a [Bayes/Lookahead, opp1, opp2, ...] eval and return seat-0 avg score / hole.

    n_players defaults to 4. The number of opponent_specs must equal
    n_players - 1. player="bayes" uses the original bayes_stage0/1,
    player="lookahead" uses the 1-step lookahead.
    """
    if len(opponent_specs) != n_players - 1:
        raise ValueError(
            f"opponent_specs length {len(opponent_specs)} != n_players - 1 = {n_players - 1}"
        )

    N = num_games
    BAYES_SEAT = 0
    tracker = BayesBeliefTracker(N, device)

    if player == "lookahead":
        s0_fn_inner = lookahead_stage0
        s1_fn_inner = lookahead_stage1
    else:
        s0_fn_inner = bayes_stage0
        s1_fn_inner = bayes_stage1

    def seat0_s0(state, pid):
        tracker.observe(state, my_player_id=pid)
        return s0_fn_inner(state, pid, tracker)

    def seat0_s1(state, pid):
        tracker.observe(state, my_player_id=pid)
        return s1_fn_inner(state, pid, tracker)

    seat_fns: List[SeatFn] = [(seat0_s0, seat0_s1)]
    for spec in opponent_specs:
        seat_fns.append(SEAT_PRESETS[spec])

    total = torch.zeros(N, dtype=torch.float32, device=device)

    for hole in range(1, holes + 1):
        state = reset_games(N, device, n_players=n_players)
        tracker.reset()
        # Observe the starting discard top so the very first decision is
        # informed.
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

                s0_fn, s1_fn = seat_fns[pid]

                state.current_stage.fill_(0)
                actions_s0 = s0_fn(state, pid)
                step_stage0(state, actions_s0, pid)
                # Observe after stage 0 too: opponent moves change discard top
                # / reveal cards. Always observe from the BAYES seat's POV.
                tracker.observe(state, my_player_id=BAYES_SEAT)
                if state.done.all():
                    break

                state.current_stage.fill_(1)
                actions_s1 = s1_fn(state, pid)
                step_stage1(state, actions_s1, pid)
                tracker.observe(state, my_player_id=BAYES_SEAT)

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        total += compute_final_score(state.player_cards[:, 0, :], device)

    return total.mean().item() / holes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument(
        "--eval-config",
        type=str,
        default="R,H,R",
        help="3 opponent seats (comma-separated). R=random, H=improved heuristic, h=base heuristic.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--player",
        type=str,
        default="bayes",
        choices=["bayes", "lookahead"],
        help="Player type: 'bayes' (belief-augmented heuristic) or 'lookahead' (1-step lookahead).",
    )
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    opponents = parse_eval_config(args.eval_config)
    n_players = 1 + len(opponents)
    label = "L" if args.player == "lookahead" else "B"

    print(f"{args.player} player: solo eval [{label},{','.join(opponents)}] "
          f"({n_players} players), {args.games} games x {args.holes} holes")
    score = run_bayes_eval(
        opponents, args.games, args.holes, device,
        n_players=n_players, player=args.player,
    )
    print(f"  {args.player} seat-0 avg score / hole: {score:.3f}")


if __name__ == "__main__":
    main()

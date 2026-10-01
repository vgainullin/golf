# Bayes lookahead (L) with one change: place the held card only when that
# lowers expected score by at least 0.5; otherwise discard and flip, which
# gains information about a hidden card.
import torch
from src.bayes_optimal import BayesBeliefTracker, lookahead_stage0, expected_score, _best_placement_score

MARGIN = 0.5
_tracker = None

def reset(state, seat):
    global _tracker
    _tracker = BayesBeliefTracker(state.player_cards.shape[0], state.player_cards.device)
    _tracker.observe(state, my_player_id=seat)

def observe(state, seat):
    _tracker.observe(state, my_player_id=seat)

def stage0(state, seat):
    _tracker.observe(state, my_player_id=seat)
    return lookahead_stage0(state, seat, _tracker)

def stage1(state, seat):
    _tracker.observe(state, my_player_id=seat)
    device = state.player_cards.device
    N = state.player_cards.shape[0]
    cards = state.player_cards[:, seat, :].clone()
    revealed = state.player_revealed[:, seat, :].clone()
    held = state.player_holding[:, seat]
    ms, tot = _tracker.multiset_by_rank(), _tracker.total()
    current_e = expected_score(cards, revealed, ms, tot, device)
    best_score, best_pos = _best_placement_score(cards, revealed, held, ms, tot, device)
    hidden = ~revealed
    idx = torch.arange(6, device=device).unsqueeze(0).expand(N, -1)
    first_hidden = torch.where(hidden, idx, torch.full((N, 6), 99, dtype=torch.long, device=device)).min(dim=1).values.clamp(0, 5)
    place = (best_score < current_e - MARGIN) | (~hidden.any(dim=1))
    return torch.where(place, 2 + best_pos, 9 + first_hidden)

# Bayes lookahead (L) with one change: when it discards and flips, flip a
# hidden card whose column partner is already revealed (that flip can complete
# a column match), instead of the first hidden slot.
import torch
from src.bayes_optimal import BayesBeliefTracker, lookahead_stage0, expected_score, _best_placement_score

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
    partner = torch.tensor([3, 4, 5, 0, 1, 2], device=device)
    partner_revealed = revealed[:, partner]
    idx = torch.arange(6, device=device).unsqueeze(0).expand(N, -1)
    big = torch.full((N, 6), 99, dtype=torch.long, device=device)
    # Prefer hidden slots with a revealed partner; fall back to any hidden slot.
    pref = torch.where(hidden & partner_revealed, idx, big).min(dim=1).values
    anyh = torch.where(hidden, idx, big).min(dim=1).values
    flip = torch.where(pref < 99, pref, anyh).clamp(0, 5)
    place = (best_score < current_e) | (~hidden.any(dim=1))
    return torch.where(place, 2 + best_pos, 9 + flip)

# The Bayes lookahead player (label L) written as agent code. This is the
# starting point the research agent edits: a run changes one thing in here.
import torch
from src.bayes_optimal import BayesBeliefTracker, lookahead_stage0, lookahead_stage1

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
    return lookahead_stage1(state, seat, _tracker)

import torch

import scripts.distill_from_bayes as distillation
from scripts.distill_from_bayes import split_train_val_by_trajectory
from scripts.distill_from_bayes import expert_tie_break_loss
from scripts.policy_audit import lookahead_stage1_scores
from src.bayes_optimal import BayesBeliefTracker, lookahead_stage1
from src.vectorized_golf import get_observation_v2, reset_games


DEVICE = torch.device("cpu")


def test_collect_expert_data_terminates_holes(monkeypatch):
    states = []
    original_reset = distillation.reset_games

    def capture_reset(*args, **kwargs):
        state = original_reset(*args, **kwargs)
        states.append(state)
        return state

    monkeypatch.setattr(distillation, "reset_games", capture_reset)
    torch.manual_seed(0)

    data = distillation.collect_expert_data(
        N=8,
        holes=1,
        device=DEVICE,
        obs_fn=get_observation_v2,
    )

    assert states[0].done.all()
    assert 0 < len(data["obs"]) < 8 * 40 * 2
    assert len(data["actions"]) == len(data["obs"])
    assert len(data["trajectory_ids"]) == len(data["obs"])


def test_score_helper_matches_production_stage1_tie_break():
    state = reset_games(1, DEVICE)
    state.player_cards[0, 0] = torch.tensor([8, 0, 11, 21, 13, 12])
    state.player_revealed[0, 0] = torch.tensor([True, True, True, True, True, False])
    state.player_holding[0, 0] = 34
    state.current_stage.fill_(1)

    tracker = BayesBeliefTracker(1, DEVICE)
    tracker.observe(state, my_player_id=0)

    production_action = lookahead_stage1(state, 0, tracker)
    helper_action, scores = lookahead_stage1_scores(state, 0, tracker)

    assert production_action.item() == 14
    assert helper_action.item() == production_action.item()
    assert scores[0, 2].item() == scores[0, 14].item()


def test_train_val_split_keeps_trajectories_disjoint():
    trajectory_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3]).numpy()

    train_idx, val_idx = split_train_val_by_trajectory(trajectory_ids, val_frac=0.25)

    train_trajectories = set(trajectory_ids[train_idx])
    val_trajectories = set(trajectory_ids[val_idx])
    assert train_trajectories
    assert val_trajectories
    assert train_trajectories.isdisjoint(val_trajectories)


def test_equal_score_expert_action_receives_tie_break_gradient():
    q = torch.zeros(1, 16, requires_grad=True)
    bl = torch.full((1, 16), float("inf"))
    valid = torch.zeros(1, 16, dtype=torch.bool)
    bl[0, 2] = 5.0
    bl[0, 14] = 5.0
    valid[0, 2] = True
    valid[0, 14] = True

    loss = expert_tie_break_loss(q, bl, valid, torch.tensor([14]))
    loss.backward()

    assert loss.item() > 0
    assert q.grad[0, 14] < 0
    assert q.grad[0, 2] > 0

"""Population-based tournament training for Golf DQN agents.

Implements a league-style evaluation system where agents play in tables
of 3 DQN + 1 random, ranked by raw average golf score. Top performers
seed the next generation.

Usage:
    python -m src.tournament --population-size 8 --generations 20 \
        --episodes-per-gen 500 --output-dir data/tournament
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from itertools import combinations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .dqn_offline import (
    GolfDQN,
    GolfDQNv2,
    GolfDQNv2Shallow,
    NUM_ACTIONS,
    STAGE0_LEGAL,
    STAGE1_LEGAL,
    mask_illegal_actions,
    resolve_device,
    set_seed,
)
from .simulation import (
    Golf,
    GolfDeck,
    Player,
    get_player_action,
)
from .tensor_dataset import tensor_to_player_tokens
from .tensor_logger import decode_action_id, encode_action_id
from .vectorized_golf import (
    VectorizedGolfState,
    reset_games,
    get_observation,
    get_observation_v2,
    step_stage0,
    step_stage1,
    compute_score as vec_compute_score,
    compute_final_score,
    get_valid_action_mask,
    heuristic_stage0,
    heuristic_stage1,
    eps_greedy_batched,
    NUM_ACTIONS as VEC_NUM_ACTIONS,
    RANK_CUTOFF,
)


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1200.0

@dataclass
class AgentRecord:
    """Metadata for one agent in the population."""
    agent_id: str
    generation: int
    elo: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_score: float = 0.0
    games_played: int = 0
    parent_id: Optional[str] = None
    elite_age: int = 0
    skip_training: bool = False
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.games_played)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / max(1, total)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["avg_score"] = round(self.avg_score, 2)
        d["win_rate"] = round(self.win_rate, 4)
        return d


# ---------------------------------------------------------------------------
# Tournament configuration
# ---------------------------------------------------------------------------

@dataclass
class TournamentConfig:
    # Population
    population_size: int = 12
    generations: int = 20
    elitism_count: int = 2          # top N agents survive unchanged

    # Training per generation
    episodes_per_gen: int = 500
    holes_per_game: int = 9
    updates_per_episode: int = 4
    batch_size: int = 256

    # Adaptive training (early generations)
    max_train_rounds: int = 5
    separation_p: float = 0.05
    adaptive_gen_limit: int = 3

    # Hyperparameter mutation
    lr_range: Tuple[float, float] = (1e-4, 3e-3)
    hidden_dim_choices: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    embedding_dim: int = 128
    gamma: float = 0.99
    mutation_rate: float = 0.3      # probability of mutating a hyperparameter
    mutation_sigma: float = 0.2     # std for log-normal LR mutation

    # Exploration
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05

    # Replay buffer (per agent)
    buffer_capacity: int = 100_000
    min_buffer_size: int = 2000

    # Target network: tau > 0 = soft Polyak update; tau = 0 = hard swap every target_update_interval
    tau: float = 0.0
    target_update_interval: int = 500

    # Match settings
    matches_per_pair: int = 4       # games per matchup in round-robin
    match_holes: int = 9
    eval_games_per_matchup: int = 20  # per C(n,3) matchup; total per agent = C(n-1,2) * this

    # Model
    model_variant: Optional[str] = None  # force variant: v1, v2, v2s (None = mixed)

    # Infrastructure
    output_dir: Path = Path("data/tournament")
    warmstart_checkpoint: Optional[Path] = None
    seed: int = 42
    device: str = "auto"
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Lightweight replay buffer (same as self_play_trainer)
# ---------------------------------------------------------------------------

import torch.nn.functional as F
from torch import nn


OBS_DIM = 30  # unified buffer width: v1 uses first 8 (rest zero-padded), v2 uses all 30


def make_model(variant: str, embedding_dim: int, hidden_dim: int, device: torch.device) -> nn.Module:
    if variant == "v2":
        return GolfDQNv2(embedding_dim, hidden_dim).to(device)
    if variant == "v2s":
        return GolfDQNv2Shallow(embedding_dim, hidden_dim).to(device)
    return GolfDQN(embedding_dim, hidden_dim).to(device)


def get_obs_fn(variant: str):
    if variant in ("v2", "v2s"):
        return get_observation_v2
    return get_observation


class ReplayBuffer:
    """Array-based circular replay buffer -- no per-transition Python objects."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self.states = np.zeros((capacity, OBS_DIM), dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, OBS_DIM), dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.stages = np.zeros(capacity, dtype=np.int64)
        self.next_stages = np.zeros(capacity, dtype=np.int64)
        self._ptr = 0
        self._size = 0

    # -- bulk insert (main hot path) --
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        stages: np.ndarray,
        next_stages: np.ndarray,
    ) -> None:
        n = len(states)
        if n == 0:
            return
        end = self._ptr + n
        if end <= self._capacity:
            self.states[self._ptr:end] = states
            self.actions[self._ptr:end] = actions
            self.rewards[self._ptr:end] = rewards
            self.next_states[self._ptr:end] = next_states
            self.dones[self._ptr:end] = dones
            self.stages[self._ptr:end] = stages
            self.next_stages[self._ptr:end] = next_stages
        else:
            first = self._capacity - self._ptr
            self.states[self._ptr:] = states[:first]
            self.actions[self._ptr:] = actions[:first]
            self.rewards[self._ptr:] = rewards[:first]
            self.next_states[self._ptr:] = next_states[:first]
            self.dones[self._ptr:] = dones[:first]
            self.stages[self._ptr:] = stages[:first]
            self.next_stages[self._ptr:] = next_stages[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_states[:rest] = next_states[first:]
            self.dones[:rest] = dones[first:]
            self.stages[:rest] = stages[first:]
            self.next_stages[:rest] = next_stages[first:]
        self._ptr = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    # -- single insert (backward compat for train_episode) --
    def push_single(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        stage: int,
        next_stage: int,
    ) -> None:
        p = self._ptr
        self.states[p] = state
        self.actions[p] = action
        self.rewards[p] = reward
        self.next_states[p] = next_state
        self.dones[p] = done
        self.stages[p] = stage
        self.next_stages[p] = next_stage
        self._ptr = (p + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    # -- sampling: returns dict of tensors on device --
    def sample(self, n: int, device: torch.device) -> Dict[str, torch.Tensor]:
        k = min(n, self._size)
        idx = np.random.choice(self._size, size=k, replace=False)
        return {
            "states": torch.as_tensor(self.states[idx], dtype=torch.long, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.long, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[idx], dtype=torch.long, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
            "stages": torch.as_tensor(self.stages[idx], dtype=torch.long, device=device),
            "next_stages": torch.as_tensor(self.next_stages[idx], dtype=torch.long, device=device),
        }

    def copy(self) -> "ReplayBuffer":
        new = ReplayBuffer(self._capacity)
        new.states[:] = self.states
        new.actions[:] = self.actions
        new.rewards[:] = self.rewards
        new.next_states[:] = self.next_states
        new.dones[:] = self.dones
        new.stages[:] = self.stages
        new.next_stages[:] = self.next_stages
        new._ptr = self._ptr
        new._size = self._size
        return new

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

def _get_tokens(golf: Golf, pid: int) -> np.ndarray:
    st = golf.encode_golf_tensor()
    cards, hold, disc = tensor_to_player_tokens(st, player_id=pid, num_players=golf.num_players)
    return np.concatenate([cards.astype(np.int64), np.array([hold, disc], dtype=np.int64)])


def _greedy_action(model: nn.Module, tokens: np.ndarray, stage: int, device: torch.device) -> int:
    st = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    sg = torch.tensor([stage], dtype=torch.long, device=device)
    mask = STAGE0_LEGAL if stage == 0 else STAGE1_LEGAL
    with torch.no_grad():
        q = model(st, sg)
    masked = q.masked_fill(~mask.to(device).unsqueeze(0), float("-inf"))
    return int(torch.argmax(masked, dim=1).item())


def _eps_greedy_action(
    model: nn.Module, tokens: np.ndarray, stage: int, eps: float, device: torch.device
) -> int:
    if random.random() < eps:
        legal = STAGE0_LEGAL if stage == 0 else STAGE1_LEGAL
        return random.choice(torch.where(legal)[0].tolist())
    return _greedy_action(model, tokens, stage, device)


def _validate_action(golf: Golf, pid: int, action_num: int, action: int, pos) -> bool:
    player = golf.players[pid]
    if action_num == 0:
        if action == 0 and len(golf.discard) == 0:
            return False
        if action == 1 and len(golf.deck) == 0:
            return False
        return action in (0, 1)
    if action_num == 1:
        if action == 0:
            return pos is not None and 0 <= pos <= 5
        if action == 1:
            if pos is None:
                return False
            row, col = divmod(pos, 3)
            return player.open_cards[row][col] == "?"
    return False


def _heuristic_action(golf: Golf, pid: int, action_num: int) -> Tuple[int, Optional[int]]:
    saved = golf.players[pid].type
    golf.players[pid].type = "Heuristic"
    try:
        action, pos, _ = get_player_action(
            deepcopy(golf), pid, action_num, rank_cutoff=4, take_random_action=False,
        )
        return action, pos
    finally:
        golf.players[pid].type = saved


def _random_action(golf: Golf, pid: int, action_num: int) -> Tuple[int, Optional[int]]:
    saved = golf.players[pid].type
    golf.players[pid].type = "Heuristic"
    try:
        action, pos, _ = get_player_action(
            deepcopy(golf), pid, action_num, take_random_action=True,
        )
        return action, pos
    finally:
        golf.players[pid].type = saved




# ---------------------------------------------------------------------------
# Self-play training episode (same core loop as self_play_trainer)
# ---------------------------------------------------------------------------

def train_episode(
    model: nn.Module,
    target_model: nn.Module,
    opponent_model: Optional[nn.Module],
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TournamentConfig,
    epsilon: float,
    global_step: int,
) -> Tuple[float, int, int]:
    """Run one self-play episode and train on buffer. Returns (loss, step, n_transitions)."""
    learner_id = 0
    n_transitions = 0

    for hole in range(1, config.holes_per_game + 1):
        players = [Player(name=f"P{i}", id=i, type="Heuristic") for i in range(4)]
        golf = Golf(players=players, deck_type="French", verbose=False)
        golf.shuffle()
        golf.deal()

        while not golf.game_over:
            for pid in range(4):
                golf.players[pid].gather_game_state(golf)
                if "?" not in golf.players[pid].game_state:
                    golf.game_over = True
                    break

                # Stage 0
                if pid == learner_id:
                    tok = _get_tokens(golf, pid)
                    aid = _eps_greedy_action(model, tok, 0, epsilon, device)
                    _, act, pos_ = decode_action_id(aid)
                    pos = None if pos_ is None else int(pos_)
                    if not _validate_action(golf, pid, 0, act, pos):
                        act, pos = _random_action(golf, pid, 0)
                        aid = encode_action_id((0, act, pos))
                elif opponent_model is not None and pid == 1:
                    tok = _get_tokens(golf, pid)
                    aid = _greedy_action(opponent_model, tok, 0, device)
                    _, act, pos_ = decode_action_id(aid)
                    pos = None if pos_ is None else int(pos_)
                    if not _validate_action(golf, pid, 0, act, pos):
                        act, pos = _heuristic_action(golf, pid, 0)
                    aid = encode_action_id((0, act, pos))
                else:
                    act, pos = _heuristic_action(golf, pid, 0)
                    aid = encode_action_id((0, act, pos))

                r0 = golf.take_action(pid, [0, act, pos])
                golf.players[pid].gather_game_state(golf)

                if pid == learner_id:
                    tok_after = _get_tokens(golf, pid)
                    buffer.push_single(
                        state=tok, action=aid, reward=float(r0),
                        next_state=tok_after,
                        done=1.0 if golf.game_over else 0.0,
                        stage=0, next_stage=1,
                    )
                    n_transitions += 1
                if golf.game_over:
                    break

                # Stage 1
                if pid == learner_id:
                    tok1 = _get_tokens(golf, pid)
                    aid1 = _eps_greedy_action(model, tok1, 1, epsilon, device)
                    _, act1, pos1_ = decode_action_id(aid1)
                    pos1 = None if pos1_ is None else int(pos1_)
                    if not _validate_action(golf, pid, 1, act1, pos1):
                        act1, pos1 = _random_action(golf, pid, 1)
                        aid1 = encode_action_id((1, act1, pos1))
                elif opponent_model is not None and pid == 1:
                    tok1 = _get_tokens(golf, pid)
                    aid1 = _greedy_action(opponent_model, tok1, 1, device)
                    _, act1, pos1_ = decode_action_id(aid1)
                    pos1 = None if pos1_ is None else int(pos1_)
                    if not _validate_action(golf, pid, 1, act1, pos1):
                        act1, pos1 = _heuristic_action(golf, pid, 1)
                else:
                    act1, pos1 = _heuristic_action(golf, pid, 1)

                r1 = golf.take_action(pid, [1, act1, pos1])
                golf.players[pid].gather_game_state(golf)

                if pid == learner_id:
                    tok1_after = _get_tokens(golf, pid)
                    buffer.push_single(
                        state=tok1, action=aid1, reward=float(r1),
                        next_state=tok1_after,
                        done=1.0 if golf.game_over else 0.0,
                        stage=1, next_stage=0,
                    )
                    n_transitions += 1

                if len(golf.deck) < golf.num_players + 2:
                    golf.deck = GolfDeck()
                    golf.shuffle()
                    golf.deal()

                if "?" not in golf.players[pid].game_state:
                    golf.last_turn = True
                    golf.end_game_player_id = pid

    # Train on buffer (transitions were pushed directly via push_single)
    losses = []
    if len(buffer) >= config.min_buffer_size:
        model.train()
        for _ in range(config.updates_per_episode):
            batch = buffer.sample(config.batch_size, device)
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_states = batch["next_states"]
            dones = batch["dones"]
            stages = batch["stages"]
            next_stages = batch["next_stages"]

            q = model(states, stages)
            q_sel = q.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                nq_online = model(next_states, next_stages)
                masked_online = mask_illegal_actions(nq_online, next_stages)
                best_next = masked_online.argmax(dim=1)
                nq_target = target_model(next_states, next_stages)
                nq_val = nq_target.gather(1, best_next.unsqueeze(1)).squeeze(1)
                nq_val = torch.where(torch.isfinite(nq_val), nq_val, torch.zeros_like(nq_val))
                targets = rewards + config.gamma * (1.0 - dones) * nq_val

            loss = F.smooth_l1_loss(q_sel, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            global_step += 1

            if config.tau > 0:
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.data.mul_(1 - config.tau).add_(p.data, alpha=config.tau)
            elif global_step % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, global_step, n_transitions


# ---------------------------------------------------------------------------
# Vectorized training and match functions
# ---------------------------------------------------------------------------

def _pad_obs(obs: torch.Tensor) -> torch.Tensor:
    """Pad observation to OBS_DIM columns if needed (v1 -> 30 with zero padding)."""
    if obs.shape[1] == OBS_DIM:
        return obs
    pad = torch.zeros(obs.shape[0], OBS_DIM - obs.shape[1], dtype=obs.dtype, device=obs.device)
    return torch.cat([obs, pad], dim=1)


def train_episodes_vectorized(
    N: int,
    model: nn.Module,
    target_model: nn.Module,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TournamentConfig,
    epsilon: float,
    global_step: int,
    obs_fn=None,
) -> Tuple[float, int]:
    """Run N episodes in parallel using mixed-table seating.

    Seat 0: Heuristic, Seat 1: Random, Seat 2: DQN learner (eps-greedy),
    Seat 3: Random. Only seat 2 records transitions.

    Args:
        obs_fn: Observation function for the learner (get_observation or get_observation_v2).
                Defaults to get_observation for backward compatibility.

    Returns (avg_loss, global_step).
    """
    if obs_fn is None:
        obs_fn = get_observation

    LEARNER_SEAT = 2

    for hole in range(1, config.holes_per_game + 1):
        state = reset_games(N, device)
        max_rounds = 30  # safety limit

        for round_num in range(max_rounds):
            if state.done.all():
                break

            for pid in range(4):
                active = ~state.done

                # If last_turn was triggered and we're back to the trigger player, mark done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                active = ~state.done

                if not active.any():
                    break

                # -- Stage 0 --
                state.current_stage.fill_(0)
                obs = obs_fn(state, pid) if pid == LEARNER_SEAT else get_observation(state, pid)

                if pid == 0:
                    # Heuristic
                    actions_s0 = heuristic_stage0(state, pid)
                elif pid == LEARNER_SEAT:
                    # DQN learner with eps-greedy
                    st = obs.to(device)
                    sg = torch.zeros(N, dtype=torch.long, device=device)
                    with torch.no_grad():
                        q = model(st, sg)
                    stage0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=device)
                    stage0_mask[:, 0] = True
                    stage0_mask[:, 1] = state.deck_ptr < 52
                    actions_s0 = eps_greedy_batched(q, epsilon, stage0_mask)
                else:
                    # Random (seats 1 and 3): eps=1.0 with dummy q-values
                    stage0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=device)
                    stage0_mask[:, 0] = True
                    stage0_mask[:, 1] = state.deck_ptr < 52
                    dummy_q = torch.zeros(N, VEC_NUM_ACTIONS, device=device)
                    actions_s0 = eps_greedy_batched(dummy_q, 1.0, stage0_mask)

                # Record stage0 transition for learner only
                if pid == LEARNER_SEAT:
                    obs_before_s0 = _pad_obs(obs).clone()
                    aid_s0 = actions_s0.clone()

                rewards_s0 = step_stage0(state, actions_s0, pid)

                if pid == LEARNER_SEAT:
                    obs_after_s0 = _pad_obs(obs_fn(state, pid))
                    active_np = active.cpu().numpy()
                    buffer.push_batch(
                        states=obs_before_s0.cpu().numpy()[active_np],
                        actions=aid_s0.cpu().numpy()[active_np],
                        rewards=np.zeros(int(active_np.sum()), dtype=np.float32),
                        next_states=obs_after_s0.cpu().numpy()[active_np],
                        dones=state.done.float().cpu().numpy()[active_np],
                        stages=np.zeros(int(active_np.sum()), dtype=np.int64),
                        next_stages=np.ones(int(active_np.sum()), dtype=np.int64),
                    )

                if state.done.all():
                    break

                # -- Stage 1 --
                state.current_stage.fill_(1)
                obs1 = obs_fn(state, pid) if pid == LEARNER_SEAT else get_observation(state, pid)

                if pid == 0:
                    # Heuristic
                    actions_s1 = heuristic_stage1(state, pid)
                elif pid == LEARNER_SEAT:
                    # DQN learner with eps-greedy
                    st1 = obs1.to(device)
                    sg1 = torch.ones(N, dtype=torch.long, device=device)
                    with torch.no_grad():
                        q1 = model(st1, sg1)
                    mask1 = get_valid_action_mask(state, pid)
                    actions_s1 = eps_greedy_batched(q1, epsilon, mask1)
                else:
                    # Random (seats 1 and 3)
                    mask1 = get_valid_action_mask(state, pid)
                    dummy_q1 = torch.zeros(N, VEC_NUM_ACTIONS, device=device)
                    actions_s1 = eps_greedy_batched(dummy_q1, 1.0, mask1)

                # Record stage1 transition for learner only
                if pid == LEARNER_SEAT:
                    obs_before_s1 = _pad_obs(obs1).clone()
                    aid_s1 = actions_s1.clone()

                rewards_s1 = step_stage1(state, actions_s1, pid)

                if pid == LEARNER_SEAT:
                    obs_after_s1 = _pad_obs(obs_fn(state, pid))
                    active_np = active.cpu().numpy()
                    buffer.push_batch(
                        states=obs_before_s1.cpu().numpy()[active_np],
                        actions=aid_s1.cpu().numpy()[active_np],
                        rewards=rewards_s1.float().cpu().numpy()[active_np],
                        next_states=obs_after_s1.cpu().numpy()[active_np],
                        dones=state.done.float().cpu().numpy()[active_np],
                        stages=np.ones(int(active_np.sum()), dtype=np.int64),
                        next_stages=np.zeros(int(active_np.sum()), dtype=np.int64),
                    )

                # Check if all cards revealed -> trigger last turn
                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

    # Train on buffer (transitions were pushed directly via push_batch)
    # Determine state width for this model variant
    is_v2 = (obs_fn is get_observation_v2)
    state_width = OBS_DIM if is_v2 else 8

    losses = []
    if len(buffer) >= config.min_buffer_size:
        model.train()
        for _ in range(config.updates_per_episode * max(1, N // 10)):
            batch = buffer.sample(config.batch_size, device)
            states = batch["states"][:, :state_width]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_states = batch["next_states"][:, :state_width]
            dones = batch["dones"]
            stages = batch["stages"]
            next_stages = batch["next_stages"]

            q = model(states, stages)
            q_sel = q.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                nq_online = model(next_states, next_stages)
                masked_online = mask_illegal_actions(nq_online, next_stages)
                best_next = masked_online.argmax(dim=1)
                nq_target = target_model(next_states, next_stages)
                nq_val = nq_target.gather(1, best_next.unsqueeze(1)).squeeze(1)
                nq_val = torch.where(torch.isfinite(nq_val), nq_val, torch.zeros_like(nq_val))
                targets = rewards + config.gamma * (1.0 - dones) * nq_val

            loss = F.smooth_l1_loss(q_sel, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            global_step += 1

            if config.tau > 0:
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.data.mul_(1 - config.tau).add_(p.data, alpha=config.tau)
            elif global_step % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, global_step



# ---------------------------------------------------------------------------
# Population manager
# ---------------------------------------------------------------------------

class TournamentTrainer:
    """Manages a population of DQN agents trained via tournament self-play."""

    def __init__(self, config: TournamentConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Population: list of (AgentRecord, model, target_model, optimizer, buffer)
        self.population: List[Tuple[AgentRecord, nn.Module, nn.Module, torch.optim.Optimizer, ReplayBuffer]] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

        # Loss tracking
        self.loss_history: Dict[str, List[float]] = {}

        # Hall of fame: best-ever agent across all generations
        self.hall_of_fame_score: float = float("inf")  # lower is better
        self.hall_of_fame_agent_id: Optional[str] = None
        self.hall_of_fame_game_scores: List[float] = []

        self._init_population()

    def _init_population(self) -> None:
        """Create initial population, optionally warm-starting from a checkpoint."""
        base_state = None
        if self.config.warmstart_checkpoint and self.config.warmstart_checkpoint.exists():
            try:
                base_state = torch.load(
                    self.config.warmstart_checkpoint, map_location="cpu", weights_only=True
                )
            except Exception:
                import pathlib
                torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
                base_state = torch.load(
                    self.config.warmstart_checkpoint, map_location="cpu", weights_only=True
                )
            print(f"Warm-starting population from {self.config.warmstart_checkpoint}")

        # If warmstarting, use the checkpoint's variant and hidden_dim so weights load,
        # and reduce LR/epsilon to avoid destroying pretrained weights
        ws_variant = None
        ws_hidden = None
        if base_state is not None:
            ws_variant = base_state.get("config", {}).get("model_variant")
            ws_hidden = base_state.get("config", {}).get("hidden_dim")
            # Cap LR and epsilon for warmstarted agents
            ws_lr_cap = 5e-4
            self.config.lr_range = (self.config.lr_range[0], min(self.config.lr_range[1], ws_lr_cap))
            self.config.epsilon_start = min(self.config.epsilon_start, 0.1)
            print(f"  Warmstart: lr_range capped to {self.config.lr_range}, epsilon_start={self.config.epsilon_start}")

        for i in range(self.config.population_size):
            # Mutate hyperparameters for diversity
            lr = np.exp(np.random.uniform(
                np.log(self.config.lr_range[0]),
                np.log(self.config.lr_range[1]),
            ))
            hidden_dim = ws_hidden if ws_hidden else random.choice(self.config.hidden_dim_choices)

            variant = ws_variant or self.config.model_variant or ("v2s" if i % 2 == 1 else "v1")

            agent_id = f"gen0_agent{i}"
            record = AgentRecord(
                agent_id=agent_id,
                generation=0,
                hyperparams={"lr": float(lr), "hidden_dim": hidden_dim, "model_variant": variant},
            )

            model = make_model(variant, self.config.embedding_dim, hidden_dim, self.device)
            target = make_model(variant, self.config.embedding_dim, hidden_dim, self.device)

            if base_state is not None:
                model.load_state_dict(base_state["model_state_dict"])
            target.load_state_dict(model.state_dict())

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            buf = ReplayBuffer(self.config.buffer_capacity)

            self.population.append((record, model, target, optimizer, buf))

        print(f"Initialized {len(self.population)} agents")
        for rec, _, _, _, _ in self.population:
            print(f"  {rec.agent_id}: lr={rec.hyperparams['lr']:.2e} hidden={rec.hyperparams['hidden_dim']} variant={rec.hyperparams.get('model_variant', 'v1')}")

    # ----- Training phase -----

    def _train_generation(self) -> None:
        """Train each agent in the population using vectorized episodes."""
        n_eps = self.config.episodes_per_gen
        progress = (self.generation - 1) / max(1, self.config.generations - 1)
        epsilon = self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
        print(f"  epsilon={epsilon:.3f}")

        for idx, (record, model, target, optimizer, buf) in enumerate(self.population):
            if record.skip_training:
                print(f"  {record.agent_id}: skip training (hall-of-fame)")
                record.skip_training = False  # only skip once
                continue

            # Children (born this generation) get higher epsilon for more exploration
            is_child = record.generation == self.generation
            agent_eps = min(epsilon + 0.15, 0.5) if is_child else epsilon

            variant = record.hyperparams.get("model_variant", "v1")
            obs_fn = get_obs_fn(variant)

            loss, _ = train_episodes_vectorized(
                N=n_eps,
                model=model,
                target_model=target,
                buffer=buf,
                optimizer=optimizer,
                device=self.device,
                config=self.config,
                epsilon=agent_eps,
                global_step=0,
                obs_fn=obs_fn,
            )

            self.loss_history.setdefault(record.agent_id, []).append(loss)
            record.hyperparams["loss"] = loss

            print(
                f"  {record.agent_id} ({variant}): trained {n_eps} vectorized eps, "
                f"buf={len(buf):,d}, loss={loss:.4f}, eps={agent_eps:.3f}"
            )

    # ----- Structured evaluation -----

    def _run_eval_config(
        self,
        seat_roles: List[str],
        model: Optional[nn.Module],
        obs_fn,
        num_games: int,
        holes: int,
    ) -> Dict[int, float]:
        """Run num_games games with specified seat roles and return avg score per hole per seat.

        seat_roles: list of 4 strings, each "dqn", "heuristic", or "random".
        model/obs_fn: used for "dqn" seats (ignored if no dqn seats).
        Returns: {seat_idx: avg_score_per_hole}.
        """
        N = num_games
        totals = {i: torch.zeros(N, dtype=torch.float32, device=self.device) for i in range(4)}

        for hole in range(1, holes + 1):
            state = reset_games(N, self.device)
            max_rounds = 30

            for round_num in range(max_rounds):
                if state.done.all():
                    break

                for pid in range(4):
                    active = ~state.done
                    back_to_trigger = state.last_turn & (state.end_game_player == pid)
                    state.done = state.done | (back_to_trigger & active)
                    active = ~state.done

                    if not active.any():
                        break

                    role = seat_roles[pid]

                    # -- Stage 0 --
                    state.current_stage.fill_(0)

                    if role == "heuristic":
                        actions_s0 = heuristic_stage0(state, pid)
                    elif role == "dqn":
                        obs = obs_fn(state, pid)
                        st = obs.to(self.device)
                        sg = torch.zeros(N, dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            q = model(st, sg)
                        s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=self.device)
                        s0_mask[:, 0] = True
                        s0_mask[:, 1] = state.deck_ptr < 52
                        actions_s0 = eps_greedy_batched(q, 0.0, s0_mask)
                    else:  # random
                        s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=self.device)
                        s0_mask[:, 0] = True
                        s0_mask[:, 1] = state.deck_ptr < 52
                        dummy_q = torch.zeros(N, VEC_NUM_ACTIONS, device=self.device)
                        actions_s0 = eps_greedy_batched(dummy_q, 1.0, s0_mask)

                    step_stage0(state, actions_s0, pid)
                    if state.done.all():
                        break

                    # -- Stage 1 --
                    state.current_stage.fill_(1)

                    if role == "heuristic":
                        actions_s1 = heuristic_stage1(state, pid)
                    elif role == "dqn":
                        obs1 = obs_fn(state, pid)
                        st1 = obs1.to(self.device)
                        sg1 = torch.ones(N, dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            q1 = model(st1, sg1)
                        mask1 = get_valid_action_mask(state, pid)
                        actions_s1 = eps_greedy_batched(q1, 0.0, mask1)
                    else:  # random
                        mask1 = get_valid_action_mask(state, pid)
                        dummy_q1 = torch.zeros(N, VEC_NUM_ACTIONS, device=self.device)
                        actions_s1 = eps_greedy_batched(dummy_q1, 1.0, mask1)

                    step_stage1(state, actions_s1, pid)

                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

            # Accumulate each seat's score for this hole
            for sid in range(4):
                hole_scores = compute_final_score(
                    state.player_cards[:, sid, :], self.device
                )
                totals[sid] += hole_scores

        return {sid: totals[sid].mean().item() / holes for sid in range(4)}, totals

    def _run_matchup_vectorized(
        self,
        dqn_entries: List[Tuple[nn.Module, Any]],
        num_games: int,
        holes: int,
    ) -> Dict[int, torch.Tensor]:
        """Play num_games with 3 DQN agents (seats 0-2) + 1 random (seat 3).

        Args:
            dqn_entries: list of 3 (model, obs_fn) tuples
            num_games: games to play (vectorized)
            holes: holes per game

        Returns: {dqn_index: tensor of per-game total scores} for indices 0,1,2
        """
        N = num_games
        totals = {sid: torch.zeros(N, dtype=torch.float32, device=self.device) for sid in range(4)}

        for hole in range(1, holes + 1):
            state = reset_games(N, self.device)
            max_rounds = 30

            for round_num in range(max_rounds):
                if state.done.all():
                    break

                for pid in range(4):
                    active = ~state.done
                    back_to_trigger = state.last_turn & (state.end_game_player == pid)
                    state.done = state.done | (back_to_trigger & active)
                    active = ~state.done

                    if not active.any():
                        break

                    # -- Stage 0 --
                    state.current_stage.fill_(0)

                    if pid < 3:  # DQN seat
                        model, obs_fn = dqn_entries[pid]
                        obs = obs_fn(state, pid)
                        st = obs.to(self.device)
                        sg = torch.zeros(N, dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            q = model(st, sg)
                        s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=self.device)
                        s0_mask[:, 0] = True
                        s0_mask[:, 1] = state.deck_ptr < 52
                        actions_s0 = eps_greedy_batched(q, 0.0, s0_mask)
                    else:  # random (seat 3)
                        s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=self.device)
                        s0_mask[:, 0] = True
                        s0_mask[:, 1] = state.deck_ptr < 52
                        dummy_q = torch.zeros(N, VEC_NUM_ACTIONS, device=self.device)
                        actions_s0 = eps_greedy_batched(dummy_q, 1.0, s0_mask)

                    step_stage0(state, actions_s0, pid)
                    if state.done.all():
                        break

                    # -- Stage 1 --
                    state.current_stage.fill_(1)

                    if pid < 3:  # DQN seat
                        model, obs_fn = dqn_entries[pid]
                        obs1 = obs_fn(state, pid)
                        st1 = obs1.to(self.device)
                        sg1 = torch.ones(N, dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            q1 = model(st1, sg1)
                        mask1 = get_valid_action_mask(state, pid)
                        actions_s1 = eps_greedy_batched(q1, 0.0, mask1)
                    else:  # random
                        mask1 = get_valid_action_mask(state, pid)
                        dummy_q1 = torch.zeros(N, VEC_NUM_ACTIONS, device=self.device)
                        actions_s1 = eps_greedy_batched(dummy_q1, 1.0, mask1)

                    step_stage1(state, actions_s1, pid)

                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

            for sid in range(4):
                hole_scores = compute_final_score(
                    state.player_cards[:, sid, :], self.device
                )
                totals[sid] += hole_scores

        return {i: totals[i] for i in range(3)}

    def _run_league_eval(self, games_per_matchup: Optional[int] = None) -> None:
        """Competitive eval: tables of 3 DQN + 1 random, all C(n,3) combinations."""
        if games_per_matchup is None:
            games_per_matchup = self.config.eval_games_per_matchup
        holes = self.config.match_holes
        n = len(self.population)

        entries = []
        for i, (rec, model, _, _, _) in enumerate(self.population):
            model.eval()
            variant = rec.hyperparams.get("model_variant", "v1")
            obs_fn = get_obs_fn(variant)
            entries.append((model, obs_fn))

        matchups = list(combinations(range(n), 3))
        agent_scores: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n)}

        print(f"  {len(matchups)} matchups, {games_per_matchup} games each, {holes} holes/game")

        for group in matchups:
            dqn_entries = [entries[i] for i in group]
            scores = self._run_matchup_vectorized(dqn_entries, games_per_matchup, holes)

            for di, agent_idx in enumerate(group):
                agent_scores[agent_idx].append(scores[di])

        for i, (rec, _, _, _, _) in enumerate(self.population):
            all_scores = torch.cat(agent_scores[i]) / holes  # per-hole avg
            avg = all_scores.mean().item()

            rec.hyperparams["avg_score"] = round(avg, 4)
            rec.hyperparams["eval_games"] = len(all_scores)
            rec.hyperparams["_game_scores"] = all_scores.tolist()

            variant = rec.hyperparams.get("model_variant", "v1")
            print(
                f"  {rec.agent_id} ({variant}): avg_score={avg:.3f} "
                f"games={len(all_scores)} "
                f"lr={rec.hyperparams['lr']:.2e} hid={rec.hyperparams['hidden_dim']}"
            )

    # ----- Selection and mutation -----

    def _select_and_mutate(self) -> None:
        """Keep elite agents, replace the rest with mutated copies of top performers."""
        n = len(self.population)

        # Rank by avg_score (lower is better)
        ranked = sorted(
            range(n),
            key=lambda i: self.population[i][0].hyperparams.get("avg_score", 999.0),
        )

        elites = ranked[: self.config.elitism_count]
        new_population = []

        # Keep elites unchanged, age-limit at 5
        for idx in elites:
            rec, model, target, opt, buf = self.population[idx]
            rec.elite_age += 1
            if rec.elite_age >= 5:
                # Force through mutation: fresh optimizer, possible LR change
                lr = rec.hyperparams["lr"]
                if random.random() < self.config.mutation_rate:
                    lr = lr * np.exp(np.random.normal(0, self.config.mutation_sigma))
                    lr = np.clip(lr, self.config.lr_range[0], self.config.lr_range[1])
                    rec.hyperparams["lr"] = float(lr)
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
                rec.elite_age = 0
            new_population.append((rec, model, target, opt, buf))

        # Fill remaining slots by mutating top half
        top_half = ranked[: max(2, len(ranked) // 2)]
        child_id = 0
        while len(new_population) < self.config.population_size:
            parent_idx = random.choice(top_half)
            parent_rec, parent_model, _, _, parent_buf = self.population[parent_idx]

            # Mutate hyperparameters (variant is inherited, never mutated)
            lr = parent_rec.hyperparams["lr"]
            hidden_dim = parent_rec.hyperparams["hidden_dim"]
            variant = parent_rec.hyperparams.get("model_variant", "v1")

            if random.random() < self.config.mutation_rate:
                lr = lr * np.exp(np.random.normal(0, self.config.mutation_sigma))
                lr = np.clip(lr, self.config.lr_range[0], self.config.lr_range[1])

            if random.random() < self.config.mutation_rate:
                hidden_dim = random.choice(self.config.hidden_dim_choices)

            agent_id = f"gen{self.generation + 1}_agent{child_id}"
            child_rec = AgentRecord(
                agent_id=agent_id,
                generation=self.generation + 1,
                parent_id=parent_rec.agent_id,
                hyperparams={"lr": float(lr), "hidden_dim": hidden_dim, "model_variant": variant},
            )

            # Copy model weights (may need new architecture if hidden_dim changed)
            child_model = make_model(variant, self.config.embedding_dim, hidden_dim, self.device)
            try:
                child_model.load_state_dict(parent_model.state_dict())
            except RuntimeError:
                pass  # Architecture changed, start fresh but keep parent's idea

            child_target = make_model(variant, self.config.embedding_dim, hidden_dim, self.device)
            child_target.load_state_dict(child_model.state_dict())
            child_opt = torch.optim.AdamW(child_model.parameters(), lr=lr, weight_decay=1e-5)
            child_buf = parent_buf.copy()

            new_population.append((child_rec, child_model, child_target, child_opt, child_buf))
            child_id += 1

        # Inject hall-of-fame agent if not already present
        hof_path = self.config.output_dir / "hall_of_fame.pt"
        if hof_path.exists() and self.hall_of_fame_agent_id is not None:
            hof_ids = {rec.agent_id for rec, _, _, _, _ in new_population}
            if self.hall_of_fame_agent_id not in hof_ids:
                hof_ckpt = torch.load(hof_path, map_location="cpu", weights_only=True)
                hof_cfg = hof_ckpt["config"]
                hof_variant = hof_cfg.get("model_variant", "v1")
                hof_hidden = hof_cfg["hidden_dim"]
                hof_embed = hof_cfg["embedding_dim"]

                hof_model = make_model(hof_variant, hof_embed, hof_hidden, self.device)
                hof_model.load_state_dict(hof_ckpt["model_state_dict"])
                hof_target = make_model(hof_variant, hof_embed, hof_hidden, self.device)
                hof_target.load_state_dict(hof_model.state_dict())

                hof_rec = AgentRecord(
                    agent_id=self.hall_of_fame_agent_id,
                    generation=hof_ckpt.get("generation", 0),
                    skip_training=True,
                    hyperparams={
                        "lr": hof_ckpt["agent_record"].get("hyperparams", {}).get("lr", 1e-3),
                        "hidden_dim": hof_hidden,
                        "model_variant": hof_variant,
                    },
                )
                lr = hof_rec.hyperparams["lr"]
                hof_opt = torch.optim.AdamW(hof_model.parameters(), lr=lr, weight_decay=1e-5)
                hof_buf = ReplayBuffer(self.config.buffer_capacity)

                # Replace the worst slot (last in new_population)
                worst_idx = max(
                    range(len(new_population)),
                    key=lambda i: new_population[i][0].hyperparams.get("avg_score", 999.0),
                )
                replaced = new_population[worst_idx][0].agent_id
                new_population[worst_idx] = (hof_rec, hof_model, hof_target, hof_opt, hof_buf)
                print(f"  Injected hall-of-fame {self.hall_of_fame_agent_id} (replacing {replaced})")

        self.population = new_population

    # ----- Checkpointing -----

    def _save_generation(self) -> None:
        from scipy import stats as sp_stats

        gen_dir = self.config.output_dir / f"gen_{self.generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        rankings = []
        for rec, model, target, _, _ in self.population:
            # Save checkpoint (exclude internal _game_scores from saved record)
            rec_dict = rec.to_dict()
            save_hp = {k: v for k, v in rec_dict.get("hyperparams", {}).items() if not k.startswith("_")}
            rec_dict["hyperparams"] = save_hp

            path = gen_dir / f"{rec.agent_id}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "target_state_dict": target.state_dict(),
                "config": {
                    "embedding_dim": self.config.embedding_dim,
                    "hidden_dim": rec.hyperparams["hidden_dim"],
                    "model_variant": rec.hyperparams.get("model_variant", "v1"),
                },
                "agent_record": rec_dict,
            }, path)
            rankings.append(rec_dict)

        # Sort by avg_score (lower is better)
        rankings.sort(key=lambda x: x.get("hyperparams", {}).get("avg_score", 999.0))

        # Save generation summary
        summary = {
            "generation": self.generation,
            "rankings": rankings,
            "best_agent": rankings[0]["agent_id"],
            "best_avg_score": rankings[0].get("hyperparams", {}).get("avg_score", "N/A"),
        }
        (gen_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))

        # Save cumulative loss history
        loss_path = self.config.output_dir / "loss_history.json"
        loss_path.write_text(json.dumps(self.loss_history, indent=2))

        # Also save the best model as the generation champion (lowest avg_score)
        best_idx = min(
            range(len(self.population)),
            key=lambda i: self.population[i][0].hyperparams.get("avg_score", 999.0),
        )
        best_model = self.population[best_idx][1]
        best_rec = self.population[best_idx][0]
        champ_path = self.config.output_dir / "champion.pt"
        champ_data = {
            "model_state_dict": best_model.state_dict(),
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": best_rec.hyperparams["hidden_dim"],
                "model_variant": best_rec.hyperparams.get("model_variant", "v1"),
            },
            "agent_record": {k: v for k, v in best_rec.to_dict().items()
                            if k != "hyperparams" or not isinstance(v, dict)} | {
                "hyperparams": {k: v for k, v in best_rec.hyperparams.items() if not k.startswith("_")},
            },
            "generation": self.generation,
        }
        torch.save(champ_data, champ_path)

        # Update hall of fame with Welch's t-test gating
        best_avg = best_rec.hyperparams.get("avg_score", 999.0)
        best_game_scores = best_rec.hyperparams.get("_game_scores", [])

        if best_avg < self.hall_of_fame_score:
            # Check statistical significance if we have a previous hall of fame
            update_hof = True
            if self.hall_of_fame_game_scores and best_game_scores:
                t_stat, p_val = sp_stats.ttest_ind(
                    best_game_scores, self.hall_of_fame_game_scores,
                    equal_var=False, alternative="less",
                )
                update_hof = p_val < 0.05
                print(
                    f"  Hall of fame t-test: t={t_stat:.3f} p={p_val:.4f} "
                    f"{'SIGNIFICANT' if update_hof else 'not significant'}"
                )

            if update_hof:
                self.hall_of_fame_score = best_avg
                self.hall_of_fame_agent_id = best_rec.agent_id
                self.hall_of_fame_game_scores = best_game_scores.copy()
                hof_path = self.config.output_dir / "hall_of_fame.pt"
                torch.save(champ_data, hof_path)
                print(f"  New hall of fame: {best_rec.agent_id} (avg_score={best_avg:.4f})")

        if self.config.hf_repo_id:
            token = self.config.hf_token or os.environ.get("HF_TOKEN")
            if token:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                api.create_repo(
                    repo_id=self.config.hf_repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                )
                api.upload_folder(
                    folder_path=str(self.config.output_dir),
                    repo_id=self.config.hf_repo_id,
                    repo_type="dataset",
                    allow_patterns=["*.json", "*.jsonl", "*.log", "champion.pt", "hall_of_fame.pt"],
                    commit_message=f"gen {self.generation}",
                    run_as_future=False,
                )
                print(f"  Uploaded gen {self.generation} results to {self.config.hf_repo_id}")
            else:
                print("  WARNING: hf_repo_id set but no token found (set HF_TOKEN env var)")

        return summary

    # ----- Main loop -----

    def run(self) -> Dict[str, Any]:
        set_seed(self.config.seed)
        start = time.time()

        import sys

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    s.write(data)
            def flush(self):
                for s in self._streams:
                    s.flush()

        _log_file = open(self.config.output_dir / "run.log", "a")
        _orig_stdout = sys.stdout
        sys.stdout = _Tee(_orig_stdout, _log_file)

        print(f"\nTournament: {self.config.generations} generations, "
              f"{self.config.population_size} agents")
        print(f"  Episodes/gen: {self.config.episodes_per_gen}")
        print(f"  Eval games/matchup: {self.config.eval_games_per_matchup}")
        print(f"  Adaptive: max_rounds={self.config.max_train_rounds} "
              f"p={self.config.separation_p} gen_limit={self.config.adaptive_gen_limit}")
        print(f"  Device: {self.device}")
        print()

        _wandb_run = None
        if self.config.wandb_project:
            import wandb
            _wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "population_size": self.config.population_size,
                    "generations": self.config.generations,
                    "episodes_per_gen": self.config.episodes_per_gen,
                    "eval_games_per_matchup": self.config.eval_games_per_matchup,
                    "elitism_count": self.config.elitism_count,
                    "lr_range": self.config.lr_range,
                    "hidden_dim_choices": self.config.hidden_dim_choices,
                    "epsilon_start": self.config.epsilon_start,
                    "epsilon_end": self.config.epsilon_end,
                },
            )

        for gen in range(1, self.config.generations + 1):
            self.generation = gen
            gen_start = time.time()

            print(f"=== Generation {gen}/{self.config.generations} ===")

            # Adaptive train+eval loop
            for round_num in range(self.config.max_train_rounds):
                print(f"Training (round {round_num + 1})...")
                self._train_generation()

                print("League evaluation...")
                self._run_league_eval()

                if gen > self.config.adaptive_gen_limit or round_num == self.config.max_train_rounds - 1:
                    break

                # Check separation between #1 and #2
                sorted_pop = sorted(
                    self.population,
                    key=lambda x: x[0].hyperparams.get("avg_score", 999.0),
                )
                scores_1 = sorted_pop[0][0].hyperparams.get("_game_scores", [])
                scores_2 = sorted_pop[1][0].hyperparams.get("_game_scores", [])

                if scores_1 and scores_2:
                    from scipy.stats import ttest_ind
                    _, p = ttest_ind(scores_1, scores_2, equal_var=False, alternative="less")
                    if p < self.config.separation_p:
                        print(f"  Clear winner after {round_num + 1} rounds (p={p:.4f})")
                        break
                    print(f"  No separation (p={p:.4f}), training round {round_num + 2}...")
                else:
                    print(f"  No scores for separation test, training round {round_num + 2}...")

            # Phase 3: Save
            summary = self._save_generation()

            gen_elapsed = time.time() - gen_start
            print(f"\nGeneration {gen} results ({gen_elapsed:.0f}s):")
            for i, r in enumerate(summary["rankings"][:5]):
                marker = " *" if i == 0 else ""
                hp = r.get('hyperparams', {})
                var = hp.get('model_variant', 'v1')
                avg_s = hp.get('avg_score', 'N/A')
                print(
                    f"  #{i+1} {r['agent_id']} ({var}): avg_score={avg_s} "
                    f"lr={hp['lr']:.2e} hid={hp['hidden_dim']}{marker}"
                )
            print()

            self.history.append(summary)

            sorted_pop = sorted(
                self.population,
                key=lambda x: x[0].hyperparams.get("avg_score", 999.0),
            )
            scores = [rec.hyperparams.get("avg_score", 999.0) for rec, *_ in sorted_pop]
            losses = [rec.hyperparams.get("loss", float("nan")) for rec, *_ in sorted_pop]
            metrics = {
                "generation": gen,
                "eval/best_score": summary["best_avg_score"],
                "eval/mean_score": float(np.mean(scores)),
                "eval/worst_score": float(np.max(scores)),
                "train/mean_loss": float(np.nanmean(losses)),
                "hof/score": self.hall_of_fame_score if self.hall_of_fame_score < float("inf") else None,
                "hof/agent_id": self.hall_of_fame_agent_id,
            }
            for rank, (loss, score) in enumerate(zip(losses, scores), 1):
                metrics[f"train/rank_{rank}_loss"] = loss
                metrics[f"eval/rank_{rank}_score"] = score

            # Always write metrics locally
            with open(self.config.output_dir / "metrics_log.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")

            if _wandb_run is not None:
                _wandb_run.log(metrics, step=gen)

            # Phase 4: Selection + Mutation (except last gen)
            if gen < self.config.generations:
                self._select_and_mutate()

        elapsed = time.time() - start
        print(f"Tournament complete: {elapsed:.0f}s")

        if _wandb_run is not None:
            _wandb_run.finish()

        sys.stdout = _orig_stdout
        _log_file.close()

        # Save history
        (self.config.output_dir / "tournament_history.json").write_text(
            json.dumps(self.history, indent=2)
        )

        return {
            "generations": self.config.generations,
            "population_size": self.config.population_size,
            "champion": self.history[-1]["best_agent"] if self.history else None,
            "champion_avg_score": self.history[-1]["best_avg_score"] if self.history else None,
            "best_ever_agent": self.hall_of_fame_agent_id,
            "best_ever_avg_score": self.hall_of_fame_score if self.hall_of_fame_score < float("inf") else None,
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> TournamentConfig:
    p = argparse.ArgumentParser(description="Tournament-style population DQN training for Golf")
    p.add_argument("--population-size", type=int, default=12)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--elitism-count", type=int, default=2)
    p.add_argument("--episodes-per-gen", type=int, default=500)
    p.add_argument("--holes-per-game", type=int, default=9)
    p.add_argument("--updates-per-episode", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--matches-per-pair", type=int, default=4)
    p.add_argument("--match-holes", type=int, default=9)
    p.add_argument("--eval-games-per-matchup", type=int, default=20)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--epsilon-start", type=float, default=0.3)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--max-train-rounds", type=int, default=5)
    p.add_argument("--separation-p", type=float, default=0.05)
    p.add_argument("--adaptive-gen-limit", type=int, default=3)
    p.add_argument("--model-variant", type=str, default=None, choices=["v1", "v2", "v2s"])
    p.add_argument("--output-dir", type=Path, default=Path("data/tournament"))
    p.add_argument("--warmstart-checkpoint", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--hf-repo-id", type=str, default=None)
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)

    args = p.parse_args(argv)
    return TournamentConfig(
        population_size=args.population_size,
        generations=args.generations,
        elitism_count=args.elitism_count,
        episodes_per_gen=args.episodes_per_gen,
        holes_per_game=args.holes_per_game,
        updates_per_episode=args.updates_per_episode,
        batch_size=args.batch_size,
        matches_per_pair=args.matches_per_pair,
        match_holes=args.match_holes,
        eval_games_per_matchup=args.eval_games_per_matchup,
        buffer_capacity=args.buffer_capacity,
        tau=args.tau,
        target_update_interval=args.target_update_interval,
        max_train_rounds=args.max_train_rounds,
        separation_p=args.separation_p,
        adaptive_gen_limit=args.adaptive_gen_limit,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        model_variant=args.model_variant,
        output_dir=args.output_dir,
        warmstart_checkpoint=args.warmstart_checkpoint,
        seed=args.seed,
        device=args.device,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def main(argv=None) -> None:
    config = parse_args(argv)
    trainer = TournamentTrainer(config)
    result = trainer.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

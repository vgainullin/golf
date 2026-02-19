"""Population-based tournament training for Golf DQN agents.

Implements an ELO-rated tournament system where a population of agents
train via self-play, compete in round-robin matches, and the top performers
seed the next generation. This creates a curriculum of increasingly strong
opponents without manual tuning.

Usage:
    python -m src.tournament --population-size 8 --generations 20 \
        --episodes-per-gen 500 --output-dir data/tournament
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .dqn_offline import (
    GolfDQN,
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


# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1200.0
ELO_K = 32.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float, rating_b: float, score_a: float
) -> Tuple[float, float]:
    """Update ELO ratings given actual score_a in [0, 1]."""
    ea = _expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    new_a = rating_a + ELO_K * (score_a - ea)
    new_b = rating_b + ELO_K * ((1.0 - score_a) - eb)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

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
    population_size: int = 8
    generations: int = 20
    elitism_count: int = 2          # top N agents survive unchanged

    # Training per generation
    episodes_per_gen: int = 500
    holes_per_game: int = 9
    updates_per_episode: int = 4
    batch_size: int = 256

    # Hyperparameter mutation
    lr_range: Tuple[float, float] = (1e-4, 3e-3)
    hidden_dim_choices: List[int] = field(default_factory=lambda: [128, 256, 512])
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

    # Target network
    target_update_interval: int = 500

    # Match settings
    matches_per_pair: int = 4       # games per matchup in round-robin
    match_holes: int = 9

    # Infrastructure
    output_dir: Path = Path("data/tournament")
    warmstart_checkpoint: Optional[Path] = None
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
# Lightweight replay buffer (same as self_play_trainer)
# ---------------------------------------------------------------------------

from collections import deque
import torch.nn.functional as F
from torch import nn


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float
    stage: int
    next_stage: int


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def extend(self, ts: List[Transition]) -> None:
        self._buf.extend(ts)

    def sample(self, n: int) -> List[Transition]:
        return random.sample(list(self._buf), min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


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
# Play a match between two models (or model vs baseline)
# ---------------------------------------------------------------------------

def play_match(
    model_a: nn.Module,
    model_b: Optional[nn.Module],
    device: torch.device,
    num_games: int = 4,
    holes: int = 9,
) -> Tuple[float, float, float, float]:
    """Play a match. model_b=None means heuristic opponent.

    Returns (score_a, score_b, wins_a, wins_b) averaged per hole.
    """
    total_a, total_b = 0.0, 0.0
    wins_a, wins_b = 0, 0

    for game_idx in range(num_games):
        # Alternate seats for fairness
        seat_a = game_idx % 4
        seat_b = (game_idx + 1) % 4

        for hole in range(1, holes + 1):
            players = []
            for i in range(4):
                players.append(Player(name=f"P{i}", id=i, type="Heuristic"))
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
                    if pid == seat_a:
                        tok = _get_tokens(golf, pid)
                        aid = _greedy_action(model_a, tok, 0, device)
                        _, act, pos_ = decode_action_id(aid)
                        pos = None if pos_ is None else int(pos_)
                        if not _validate_action(golf, pid, 0, act, pos):
                            act, pos = _heuristic_action(golf, pid, 0)
                    elif pid == seat_b and model_b is not None:
                        tok = _get_tokens(golf, pid)
                        aid = _greedy_action(model_b, tok, 0, device)
                        _, act, pos_ = decode_action_id(aid)
                        pos = None if pos_ is None else int(pos_)
                        if not _validate_action(golf, pid, 0, act, pos):
                            act, pos = _heuristic_action(golf, pid, 0)
                    else:
                        act, pos = _heuristic_action(golf, pid, 0)

                    golf.take_action(pid, [0, act, pos])
                    golf.players[pid].gather_game_state(golf)
                    if golf.game_over:
                        break

                    # Stage 1
                    if pid == seat_a:
                        tok = _get_tokens(golf, pid)
                        aid = _greedy_action(model_a, tok, 1, device)
                        _, act1, pos1_ = decode_action_id(aid)
                        pos1 = None if pos1_ is None else int(pos1_)
                        if not _validate_action(golf, pid, 1, act1, pos1):
                            act1, pos1 = _heuristic_action(golf, pid, 1)
                    elif pid == seat_b and model_b is not None:
                        tok = _get_tokens(golf, pid)
                        aid = _greedy_action(model_b, tok, 1, device)
                        _, act1, pos1_ = decode_action_id(aid)
                        pos1 = None if pos1_ is None else int(pos1_)
                        if not _validate_action(golf, pid, 1, act1, pos1):
                            act1, pos1 = _heuristic_action(golf, pid, 1)
                    else:
                        act1, pos1 = _heuristic_action(golf, pid, 1)

                    golf.take_action(pid, [1, act1, pos1])
                    golf.players[pid].gather_game_state(golf)

                    if len(golf.deck) < golf.num_players + 2:
                        golf.deck = GolfDeck()
                        golf.shuffle()
                        golf.deal()

                    if "?" not in golf.players[pid].game_state:
                        golf.last_turn = True
                        golf.end_game_player_id = pid

            for p in golf.players:
                p.calculate_score(final=True)
            total_a += golf.players[seat_a].score
            total_b += golf.players[seat_b].score

    n = num_games * holes
    avg_a = total_a / max(1, n)
    avg_b = total_b / max(1, n)
    # In golf, lower is better - count wins by who scored lower
    wins_a = 1 if avg_a < avg_b else 0
    wins_b = 1 if avg_b < avg_a else 0
    return avg_a, avg_b, float(wins_a), float(wins_b)


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
) -> Tuple[float, int, List[Transition]]:
    """Run one self-play episode and train on buffer. Returns (loss, step, transitions)."""
    learner_id = 0
    transitions: List[Transition] = []

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
                    transitions.append(Transition(
                        state=tok, action=aid, reward=float(r0),
                        next_state=tok_after,
                        done=1.0 if golf.game_over else 0.0,
                        stage=0, next_stage=1,
                    ))
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
                    transitions.append(Transition(
                        state=tok1, action=aid1, reward=float(r1),
                        next_state=tok1_after,
                        done=1.0 if golf.game_over else 0.0,
                        stage=1, next_stage=0,
                    ))

                if len(golf.deck) < golf.num_players + 2:
                    golf.deck = GolfDeck()
                    golf.shuffle()
                    golf.deal()

                if "?" not in golf.players[pid].game_state:
                    golf.last_turn = True
                    golf.end_game_player_id = pid

    # Add transitions to buffer
    buffer.extend(transitions)

    # Train on buffer
    losses = []
    if len(buffer) >= config.min_buffer_size:
        model.train()
        for _ in range(config.updates_per_episode):
            batch = buffer.sample(config.batch_size)
            states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.long, device=device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
            next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.long, device=device)
            dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
            stages = torch.tensor([t.stage for t in batch], dtype=torch.long, device=device)
            next_stages = torch.tensor([t.next_stage for t in batch], dtype=torch.long, device=device)

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

            if global_step % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, global_step, transitions


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

        for i in range(self.config.population_size):
            # Mutate hyperparameters for diversity
            lr = np.exp(np.random.uniform(
                np.log(self.config.lr_range[0]),
                np.log(self.config.lr_range[1]),
            ))
            hidden_dim = random.choice(self.config.hidden_dim_choices)

            agent_id = f"gen0_agent{i}"
            record = AgentRecord(
                agent_id=agent_id,
                generation=0,
                hyperparams={"lr": float(lr), "hidden_dim": hidden_dim},
            )

            model = GolfDQN(self.config.embedding_dim, hidden_dim).to(self.device)
            target = GolfDQN(self.config.embedding_dim, hidden_dim).to(self.device)

            if base_state is not None:
                # Only load if architecture matches
                try:
                    model.load_state_dict(base_state["model_state_dict"])
                except RuntimeError:
                    pass  # Architecture mismatch, start from scratch
            target.load_state_dict(model.state_dict())

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            buf = ReplayBuffer(self.config.buffer_capacity)

            self.population.append((record, model, target, optimizer, buf))

        print(f"Initialized {len(self.population)} agents")
        for rec, _, _, _, _ in self.population:
            print(f"  {rec.agent_id}: lr={rec.hyperparams['lr']:.2e} hidden={rec.hyperparams['hidden_dim']}")

    # ----- Training phase -----

    def _train_generation(self) -> None:
        """Train each agent in the population for episodes_per_gen episodes."""
        eps_start = self.config.epsilon_start
        eps_end = self.config.epsilon_end
        n_eps = self.config.episodes_per_gen

        for idx, (record, model, target, optimizer, buf) in enumerate(self.population):
            global_step = 0
            log_interval = max(1, n_eps // 10)
            for ep in range(1, n_eps + 1):
                progress = ep / n_eps
                epsilon = eps_start + progress * (eps_end - eps_start)

                # Pick a random opponent from the population (not self)
                opp_indices = [j for j in range(len(self.population)) if j != idx]
                opp_idx = random.choice(opp_indices)
                opp_model = self.population[opp_idx][1]
                opp_model.eval()

                loss, global_step, _ = train_episode(
                    model, target, opp_model, buf, optimizer,
                    self.device, self.config, epsilon, global_step,
                )

                if ep % log_interval == 0:
                    print(
                        f"  {record.agent_id}: ep {ep}/{n_eps}, "
                        f"buf={len(buf):,d}, loss={loss:.4f}, eps={epsilon:.3f}"
                    )

            print(
                f"  {record.agent_id}: trained {n_eps} eps, "
                f"buf={len(buf):,d}, loss={loss:.4f}"
            )

    # ----- Round-robin tournament -----

    def _run_tournament(self) -> None:
        """Round-robin matches between all agents. Updates ELO and stats."""
        n = len(self.population)
        for i in range(n):
            for j in range(i + 1, n):
                rec_a, model_a, _, _, _ = self.population[i]
                rec_b, model_b, _, _, _ = self.population[j]
                model_a.eval()
                model_b.eval()

                score_a, score_b, wa, wb = play_match(
                    model_a, model_b, self.device,
                    num_games=self.config.matches_per_pair,
                    holes=self.config.match_holes,
                )

                # In golf, lower score wins
                if score_a < score_b:
                    actual_a = 1.0
                elif score_a > score_b:
                    actual_a = 0.0
                else:
                    actual_a = 0.5

                rec_a.elo, rec_b.elo = update_elo(rec_a.elo, rec_b.elo, actual_a)

                rec_a.total_score += score_a * self.config.matches_per_pair * self.config.match_holes
                rec_b.total_score += score_b * self.config.matches_per_pair * self.config.match_holes
                rec_a.games_played += self.config.matches_per_pair
                rec_b.games_played += self.config.matches_per_pair

                if actual_a > 0.5:
                    rec_a.wins += 1
                    rec_b.losses += 1
                elif actual_a < 0.5:
                    rec_b.wins += 1
                    rec_a.losses += 1
                else:
                    rec_a.draws += 1
                    rec_b.draws += 1

        # Also play each agent vs heuristic baseline for absolute measurement
        for rec, model, _, _, _ in self.population:
            model.eval()
            score_agent, score_heur, _, _ = play_match(
                model, None, self.device,
                num_games=self.config.matches_per_pair,
                holes=self.config.match_holes,
            )
            rec.hyperparams["vs_heuristic_score"] = round(score_agent, 2)

    # ----- Selection and mutation -----

    def _select_and_mutate(self) -> None:
        """Keep elite agents, replace the rest with mutated copies of top performers."""
        # Sort by ELO (descending)
        ranked = sorted(
            range(len(self.population)),
            key=lambda i: self.population[i][0].elo,
            reverse=True,
        )

        elites = ranked[: self.config.elitism_count]
        new_population = []

        # Keep elites unchanged
        for idx in elites:
            rec, model, target, opt, buf = self.population[idx]
            new_population.append((rec, model, target, opt, buf))

        # Fill remaining slots by mutating top half
        top_half = ranked[: max(1, len(ranked) // 2)]
        child_id = 0
        while len(new_population) < self.config.population_size:
            parent_idx = random.choice(top_half)
            parent_rec, parent_model, _, _, _ = self.population[parent_idx]

            # Mutate hyperparameters
            lr = parent_rec.hyperparams["lr"]
            hidden_dim = parent_rec.hyperparams["hidden_dim"]

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
                hyperparams={"lr": float(lr), "hidden_dim": hidden_dim},
            )

            # Copy model weights (may need new architecture if hidden_dim changed)
            child_model = GolfDQN(self.config.embedding_dim, hidden_dim).to(self.device)
            try:
                child_model.load_state_dict(parent_model.state_dict())
            except RuntimeError:
                pass  # Architecture changed, start fresh but keep parent's idea

            child_target = GolfDQN(self.config.embedding_dim, hidden_dim).to(self.device)
            child_target.load_state_dict(child_model.state_dict())
            child_opt = torch.optim.AdamW(child_model.parameters(), lr=lr, weight_decay=1e-5)
            child_buf = ReplayBuffer(self.config.buffer_capacity)

            new_population.append((child_rec, child_model, child_target, child_opt, child_buf))
            child_id += 1

        self.population = new_population

    # ----- Checkpointing -----

    def _save_generation(self) -> None:
        gen_dir = self.config.output_dir / f"gen_{self.generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        rankings = []
        for rec, model, target, _, _ in self.population:
            # Save checkpoint
            path = gen_dir / f"{rec.agent_id}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "target_state_dict": target.state_dict(),
                "config": {
                    "embedding_dim": self.config.embedding_dim,
                    "hidden_dim": rec.hyperparams["hidden_dim"],
                },
                "agent_record": rec.to_dict(),
            }, path)
            rankings.append(rec.to_dict())

        # Sort by ELO
        rankings.sort(key=lambda x: x["elo"], reverse=True)

        # Save generation summary
        summary = {
            "generation": self.generation,
            "rankings": rankings,
            "best_agent": rankings[0]["agent_id"],
            "best_elo": rankings[0]["elo"],
            "best_vs_heuristic": rankings[0].get("hyperparams", {}).get("vs_heuristic_score", "N/A"),
        }
        (gen_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))

        # Also save the best model as the generation champion
        best_idx = max(
            range(len(self.population)),
            key=lambda i: self.population[i][0].elo,
        )
        best_model = self.population[best_idx][1]
        best_rec = self.population[best_idx][0]
        champ_path = self.config.output_dir / "champion.pt"
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": best_rec.hyperparams["hidden_dim"],
            },
            "agent_record": best_rec.to_dict(),
            "generation": self.generation,
        }, champ_path)

        return summary

    # ----- Main loop -----

    def run(self) -> Dict[str, Any]:
        set_seed(self.config.seed)
        start = time.time()

        print(f"\nTournament: {self.config.generations} generations, "
              f"{self.config.population_size} agents")
        print(f"  Episodes/gen: {self.config.episodes_per_gen}")
        print(f"  Matches/pair: {self.config.matches_per_pair}")
        print(f"  Device: {self.device}")
        print()

        for gen in range(1, self.config.generations + 1):
            self.generation = gen
            gen_start = time.time()

            print(f"=== Generation {gen}/{self.config.generations} ===")

            # Phase 1: Train
            print("Training...")
            self._train_generation()

            # Phase 2: Tournament
            print("Tournament round-robin...")
            self._run_tournament()

            # Phase 3: Save
            summary = self._save_generation()

            gen_elapsed = time.time() - gen_start
            print(f"\nGeneration {gen} results ({gen_elapsed:.0f}s):")
            for i, r in enumerate(summary["rankings"][:5]):
                marker = " *" if i == 0 else ""
                print(
                    f"  #{i+1} {r['agent_id']}: ELO={r['elo']:.0f} "
                    f"score={r['avg_score']} wr={r['win_rate']:.1%} "
                    f"lr={r['hyperparams']['lr']:.2e} "
                    f"hid={r['hyperparams']['hidden_dim']}{marker}"
                )
            print()

            self.history.append(summary)

            # Phase 4: Selection + Mutation (except last gen)
            if gen < self.config.generations:
                self._select_and_mutate()

        elapsed = time.time() - start
        print(f"Tournament complete: {elapsed:.0f}s")

        # Save history
        (self.config.output_dir / "tournament_history.json").write_text(
            json.dumps(self.history, indent=2)
        )

        return {
            "generations": self.config.generations,
            "population_size": self.config.population_size,
            "champion": self.history[-1]["best_agent"] if self.history else None,
            "champion_elo": self.history[-1]["best_elo"] if self.history else None,
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> TournamentConfig:
    p = argparse.ArgumentParser(description="Tournament-style population DQN training for Golf")
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--elitism-count", type=int, default=2)
    p.add_argument("--episodes-per-gen", type=int, default=500)
    p.add_argument("--holes-per-game", type=int, default=9)
    p.add_argument("--updates-per-episode", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--matches-per-pair", type=int, default=4)
    p.add_argument("--match-holes", type=int, default=9)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--epsilon-start", type=float, default=0.3)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=Path("data/tournament"))
    p.add_argument("--warmstart-checkpoint", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

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
        buffer_capacity=args.buffer_capacity,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        output_dir=args.output_dir,
        warmstart_checkpoint=args.warmstart_checkpoint,
        seed=args.seed,
        device=args.device,
    )


def main(argv=None) -> None:
    config = parse_args(argv)
    trainer = TournamentTrainer(config)
    result = trainer.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

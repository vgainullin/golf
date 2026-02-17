"""Online self-play DQN training for Golf.

Implements an ε-greedy self-play loop with experience replay, target network
updates, opponent pool management, and periodic evaluation against baselines.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dqn_offline import (
    GolfDQN,
    NUM_ACTIONS,
    STAGE0_LEGAL,
    STAGE1_LEGAL,
    STATE_SEQUENCE_LENGTH,
    mask_illegal_actions,
    resolve_device,
    set_seed,
)
from .simulation import (
    Golf,
    GolfDeck,
    Player,
    calc_opt_heuristic_position,
    encode_pos_tuple,
    get_player_action,
)
from .tensor_dataset import tensor_to_player_tokens
from .tensor_logger import encode_action_id, decode_action_id


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray       # (8,) int64 tokens
    action: int             # action id 0-15
    reward: float
    next_state: np.ndarray  # (8,) int64 tokens
    done: float             # 0.0 or 1.0
    stage: int              # 0 or 1
    next_stage: int         # 0 or 1


class ReplayBuffer:
    """Fixed-capacity circular replay buffer with uniform sampling."""

    def __init__(self, capacity: int):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def push_many(self, transitions: List[Transition]) -> None:
        self._buffer.extend(transitions)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Maintains a pool of historical checkpoints for opponent diversity."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._checkpoints: List[Dict[str, Any]] = []

    def add(self, state_dict: dict, episode: int, score: float) -> None:
        entry = {
            "state_dict": {k: v.cpu().clone() for k, v in state_dict.items()},
            "episode": episode,
            "score": score,
        }
        self._checkpoints.append(entry)
        if len(self._checkpoints) > self.max_size:
            self._checkpoints.pop(0)

    def sample(self) -> Optional[dict]:
        if not self._checkpoints:
            return None
        return random.choice(self._checkpoints)["state_dict"]

    def __len__(self) -> int:
        return len(self._checkpoints)


# ---------------------------------------------------------------------------
# Self-play configuration
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    # Training loop
    num_episodes: int = 5000
    holes_per_game: int = 9
    updates_per_episode: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    grad_clip: float = 1.0
    weight_decay: float = 1e-5

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3000

    # Replay buffer
    buffer_capacity: int = 200_000
    min_buffer_size: int = 5000

    # Target network
    target_update_interval: int = 500  # in gradient steps

    # Opponents: "random", "heuristic", "self", "pool"
    opponent_types: List[str] = field(default_factory=lambda: ["heuristic", "self", "pool"])

    # Model architecture
    embedding_dim: int = 128
    hidden_dim: int = 256

    # Evaluation
    eval_interval: int = 250  # episodes between evaluations
    eval_games: int = 100

    # Checkpointing
    checkpoint_interval: int = 500
    output_dir: Path = Path("data/self_play")

    # Opponent pool
    pool_save_interval: int = 500
    pool_max_size: int = 10

    # Warm start
    warmstart_checkpoint: Optional[Path] = None

    # Misc
    seed: int = 42
    device: str = "auto"
    learner_seat: int = 0  # which seat the learner occupies


# ---------------------------------------------------------------------------
# Helper: get tokens from game state for a player
# ---------------------------------------------------------------------------

def _get_player_tokens(golf: Golf, player_id: int) -> np.ndarray:
    """Extract 8-token state vector for a player from the current game state."""
    state_tensor = golf.encode_golf_tensor()
    cards, holding, discard_top = tensor_to_player_tokens(
        state_tensor, player_id=player_id, num_players=golf.num_players,
    )
    return np.concatenate([
        cards.astype(np.int64, copy=False),
        np.array([holding, discard_top], dtype=np.int64),
    ])


def _get_legal_actions(stage: int) -> torch.Tensor:
    """Return boolean mask of legal actions for a stage."""
    if stage == 0:
        return STAGE0_LEGAL
    return STAGE1_LEGAL


# ---------------------------------------------------------------------------
# Self-play trainer
# ---------------------------------------------------------------------------

class SelfPlayTrainer:
    """Orchestrates online self-play DQN training."""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.device = resolve_device(config.device)

        # Models
        self.model = GolfDQN(config.embedding_dim, config.hidden_dim).to(self.device)
        self.target_model = GolfDQN(config.embedding_dim, config.hidden_dim).to(self.device)

        # Warm-start from checkpoint
        if config.warmstart_checkpoint and config.warmstart_checkpoint.exists():
            try:
                state = torch.load(config.warmstart_checkpoint, map_location="cpu", weights_only=True)
            except Exception:
                import pathlib
                torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
                state = torch.load(config.warmstart_checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"Warm-started from {config.warmstart_checkpoint}")

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Replay buffer and opponent pool
        self.buffer = ReplayBuffer(config.buffer_capacity)
        self.opponent_pool = OpponentPool(config.pool_max_size)

        # Counters
        self.global_step = 0
        self.episode = 0

        # Action masks on device
        self._stage0_mask = STAGE0_LEGAL.to(self.device)
        self._stage1_mask = STAGE1_LEGAL.to(self.device)

        # Opponent DQN model (for "self" and "pool" opponents)
        self._opponent_model = GolfDQN(config.embedding_dim, config.hidden_dim).to(self.device)
        self._opponent_model.load_state_dict(self.model.state_dict())
        self._opponent_model.eval()

    # ----- Epsilon schedule -----

    def _get_epsilon(self) -> float:
        progress = min(1.0, self.episode / max(1, self.config.epsilon_decay_episodes))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    # ----- Action selection -----

    def _select_action_greedy(
        self, model: nn.Module, tokens: np.ndarray, stage: int
    ) -> int:
        state_t = torch.as_tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        stage_t = torch.tensor([stage], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = model(state_t, stage_t)
        mask = self._stage0_mask if stage == 0 else self._stage1_mask
        masked_q = q.masked_fill(~mask.unsqueeze(0), float("-inf"))
        return int(torch.argmax(masked_q, dim=1).item())

    def _select_action_epsilon_greedy(self, tokens: np.ndarray, stage: int, epsilon: float) -> int:
        if random.random() < epsilon:
            legal = _get_legal_actions(stage)
            legal_indices = torch.where(legal)[0].tolist()
            return random.choice(legal_indices)
        return self._select_action_greedy(self.model, tokens, stage)

    # ----- Opponent action -----

    def _get_opponent_action(
        self, golf: Golf, player_id: int, action_num: int, opponent_type: str
    ) -> Tuple[int, Optional[int]]:
        """Return (action, position) for an opponent player."""
        # get_player_action checks player.type; temporarily set to Heuristic
        # so the function's internal logic works for both random and heuristic calls.
        saved_type = golf.players[player_id].type
        golf.players[player_id].type = "Heuristic"

        try:
            if opponent_type == "random":
                action, pos, _ = get_player_action(
                    deepcopy(golf), player_id, action_num, take_random_action=True,
                )
                return action, pos

            if opponent_type == "heuristic":
                action, pos, _ = get_player_action(
                    deepcopy(golf), player_id, action_num, rank_cutoff=4, take_random_action=False,
                )
                return action, pos

            if opponent_type in ("self", "pool"):
                tokens = _get_player_tokens(golf, player_id)
                action_id = self._select_action_greedy(self._opponent_model, tokens, action_num)
                sel_action_num, action, position = decode_action_id(action_id)
                if sel_action_num != action_num:
                    action, pos, _ = get_player_action(
                        deepcopy(golf), player_id, action_num, take_random_action=True,
                    )
                    return action, pos
                pos = None if position is None else int(position)
                if not self._validate_action(golf, player_id, action_num, action, pos):
                    action, pos, _ = get_player_action(
                        deepcopy(golf), player_id, action_num, take_random_action=True,
                    )
                return action, pos

            # Fallback
            action, pos, _ = get_player_action(
                deepcopy(golf), player_id, action_num, take_random_action=True,
            )
            return action, pos
        finally:
            golf.players[player_id].type = saved_type

    def _validate_action(
        self, golf: Golf, player_id: int, action_num: int, action: int, pos: Optional[int]
    ) -> bool:
        """Check if an action is valid in the current game state."""
        player = golf.players[player_id]
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

    # ----- Play one episode -----

    def _pick_opponent_type(self) -> str:
        """Pick an opponent type from the configured mix."""
        choices = list(self.config.opponent_types)
        # If pool is requested but empty, fall back to self
        if "pool" in choices and len(self.opponent_pool) == 0:
            choices = [c if c != "pool" else "self" for c in choices]
        return random.choice(choices)

    def _play_episode(self, epsilon: float) -> Tuple[List[Transition], float]:
        """Play a full game and return transitions for the learner + final score."""
        learner_id = self.config.learner_seat
        opponent_type = self._pick_opponent_type()

        transitions: List[Transition] = []
        total_reward = 0.0

        for hole in range(1, self.config.holes_per_game + 1):
            # Create fresh players and game
            players = []
            for i in range(4):
                if i == learner_id:
                    players.append(Player(name=f"Learner", id=i, type="OfflineDQN"))
                elif i % 2 == 1:
                    players.append(Player(name=f"Opp_{i}", id=i, type="Heuristic"))
                else:
                    players.append(Player(name=f"Opp_{i}", id=i, type="Random"))

            golf = Golf(players=players, deck_type="French", verbose=False)
            golf.shuffle()
            golf.deal()

            round_num = 0
            while not golf.game_over:
                for pid in range(golf.num_players):
                    golf.players[pid].gather_game_state(golf)

                    if "?" not in golf.players[pid].game_state:
                        golf.game_over = True
                        break

                    # --- Stage 0: Draw ---
                    if pid == learner_id:
                        tokens_before = _get_player_tokens(golf, pid)
                        action_id = self._select_action_epsilon_greedy(tokens_before, 0, epsilon)
                        a_num, action, position = decode_action_id(action_id)
                        pos = None if position is None else int(position)
                        if not self._validate_action(golf, pid, 0, action, pos):
                            action, pos, _ = get_player_action(
                                deepcopy(golf), pid, 0, take_random_action=True,
                            )
                            action_id = encode_action_id((0, action, pos))
                    else:
                        action, pos = self._get_opponent_action(golf, pid, 0, opponent_type)
                        action_id = encode_action_id((0, action, pos))

                    reward_0 = golf.take_action(pid, [0, action, pos])
                    golf.players[pid].gather_game_state(golf)

                    if pid == learner_id:
                        tokens_after_0 = _get_player_tokens(golf, pid)
                        transitions.append(Transition(
                            state=tokens_before,
                            action=action_id,
                            reward=float(reward_0),
                            next_state=tokens_after_0,
                            done=1.0 if golf.game_over else 0.0,
                            stage=0,
                            next_stage=1,
                        ))
                        total_reward += reward_0

                    if golf.game_over:
                        break

                    # --- Stage 1: Place/Discard ---
                    if pid == learner_id:
                        tokens_before_1 = _get_player_tokens(golf, pid)
                        action_id_1 = self._select_action_epsilon_greedy(tokens_before_1, 1, epsilon)
                        a_num_1, action_1, position_1 = decode_action_id(action_id_1)
                        pos_1 = None if position_1 is None else int(position_1)
                        if not self._validate_action(golf, pid, 1, action_1, pos_1):
                            action_1, pos_1, _ = get_player_action(
                                deepcopy(golf), pid, 1, take_random_action=True,
                            )
                            action_id_1 = encode_action_id((1, action_1, pos_1))
                    else:
                        action_1, pos_1 = self._get_opponent_action(golf, pid, 1, opponent_type)
                        action_id_1 = encode_action_id((1, action_1, pos_1))

                    reward_1 = golf.take_action(pid, [1, action_1, pos_1])
                    golf.players[pid].gather_game_state(golf)

                    if pid == learner_id:
                        tokens_after_1 = _get_player_tokens(golf, pid)
                        transitions.append(Transition(
                            state=tokens_before_1,
                            action=action_id_1,
                            reward=float(reward_1),
                            next_state=tokens_after_1,
                            done=1.0 if golf.game_over else 0.0,
                            stage=1,
                            next_stage=0,
                        ))
                        total_reward += reward_1

                    # Deck replenish
                    if len(golf.deck) < golf.num_players + 2:
                        golf.deck = GolfDeck()
                        golf.shuffle()
                        golf.deal()

                    if "?" not in golf.players[pid].game_state:
                        golf.last_turn = True
                        golf.end_game_player_id = pid

                round_num += 1

            # Calculate final score for learner
            golf.players[learner_id].calculate_score(final=True)

        final_score = golf.players[learner_id].score if golf.players else 0.0
        return transitions, final_score

    # ----- Training step -----

    def _train_batch(self) -> float:
        """Sample a batch from the buffer and perform one DQN update."""
        batch = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.long, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.long, device=self.device
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )
        stages = torch.tensor(
            [t.stage for t in batch], dtype=torch.long, device=self.device
        )
        next_stages = torch.tensor(
            [t.next_stage for t in batch], dtype=torch.long, device=self.device
        )

        # Double DQN: use online model to select action, target to evaluate
        self.model.train()
        q_values = self.model(states, stages)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Online model selects best next action
            next_q_online = self.model(next_states, next_stages)
            masked_online = mask_illegal_actions(next_q_online, next_stages)
            best_next_actions = masked_online.argmax(dim=1)

            # Target model evaluates that action
            next_q_target = self.target_model(next_states, next_stages)
            next_q_val = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            next_q_val = torch.where(
                torch.isfinite(next_q_val), next_q_val, torch.zeros_like(next_q_val)
            )
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_val

        loss = F.smooth_l1_loss(q_selected, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.global_step += 1

        # Target network sync
        if self.global_step % self.config.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())

    # ----- Evaluation -----

    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate the current policy greedily against baseline opponents."""
        self.model.eval()
        learner_id = self.config.learner_seat
        total_score = 0.0
        wins = 0
        game_scores: List[float] = []

        for game in range(self.config.eval_games):
            game_total = 0.0
            for hole in range(1, self.config.holes_per_game + 1):
                players = [
                    Player(name="Learner", id=0, type="OfflineDQN"),
                    Player(name="Heuristic_1", id=1, type="Heuristic"),
                    Player(name="Random_2", id=2, type="Random"),
                    Player(name="Heuristic_3", id=3, type="Heuristic"),
                ]
                golf = Golf(players=players, deck_type="French", verbose=False)
                golf.shuffle()
                golf.deal()

                round_num = 0
                while not golf.game_over:
                    for pid in range(golf.num_players):
                        golf.players[pid].gather_game_state(golf)
                        if "?" not in golf.players[pid].game_state:
                            golf.game_over = True
                            break

                        # Stage 0
                        if pid == learner_id:
                            tokens = _get_player_tokens(golf, pid)
                            action_id = self._select_action_greedy(self.model, tokens, 0)
                            _, action, position = decode_action_id(action_id)
                            pos = None if position is None else int(position)
                            if not self._validate_action(golf, pid, 0, action, pos):
                                action, pos, _ = get_player_action(
                                    deepcopy(golf), pid, 0, take_random_action=True,
                                )
                        else:
                            ptype = golf.players[pid].type
                            take_random = ptype == "Random"
                            action, pos, _ = get_player_action(
                                deepcopy(golf), pid, 0, rank_cutoff=4, take_random_action=take_random,
                            )

                        golf.take_action(pid, [0, action, pos])
                        golf.players[pid].gather_game_state(golf)
                        if golf.game_over:
                            break

                        # Stage 1
                        if pid == learner_id:
                            tokens = _get_player_tokens(golf, pid)
                            action_id = self._select_action_greedy(self.model, tokens, 1)
                            _, action_1, position_1 = decode_action_id(action_id)
                            pos_1 = None if position_1 is None else int(position_1)
                            if not self._validate_action(golf, pid, 1, action_1, pos_1):
                                action_1, pos_1, _ = get_player_action(
                                    deepcopy(golf), pid, 1, take_random_action=True,
                                )
                        else:
                            ptype = golf.players[pid].type
                            take_random = ptype == "Random"
                            action_1, pos_1, _ = get_player_action(
                                deepcopy(golf), pid, 1, rank_cutoff=4, take_random_action=take_random,
                            )

                        golf.take_action(pid, [1, action_1, pos_1])
                        golf.players[pid].gather_game_state(golf)

                        if len(golf.deck) < golf.num_players + 2:
                            golf.deck = GolfDeck()
                            golf.shuffle()
                            golf.deal()

                        if "?" not in golf.players[pid].game_state:
                            golf.last_turn = True
                            golf.end_game_player_id = pid

                    round_num += 1

                for p in golf.players:
                    p.calculate_score(final=True)

                game_total += golf.players[learner_id].score

            game_scores.append(game_total / self.config.holes_per_game)
            # Check if learner won (lowest total score)
            all_scores = [p.score for p in golf.players]
            if golf.players[learner_id].score <= min(all_scores):
                wins += 1

        avg_score = float(np.mean(game_scores))
        std_score = float(np.std(game_scores))
        win_rate = wins / max(1, self.config.eval_games)

        return {
            "avg_score": round(avg_score, 2),
            "std_score": round(std_score, 2),
            "win_rate": round(win_rate, 4),
            "games": self.config.eval_games,
        }

    # ----- Checkpoint -----

    def _save_checkpoint(self, tag: str = "latest") -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / f"self_play_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_state_dict": self.target_model.state_dict(),
                "config": asdict(self.config),
                "episode": self.episode,
                "global_step": self.global_step,
            },
            path,
        )
        return path

    # ----- Main training loop -----

    def train(self) -> Dict[str, Any]:
        """Run the full self-play training loop."""
        set_seed(self.config.seed)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        history: List[Dict[str, Any]] = []
        best_eval_score = float("inf")
        start_time = time.time()

        print(f"Self-play training: {self.config.num_episodes} episodes")
        print(f"  Buffer capacity: {self.config.buffer_capacity}")
        print(f"  Opponents: {self.config.opponent_types}")
        print(f"  Warm-start: {self.config.warmstart_checkpoint}")
        print(f"  Device: {self.device}")
        print()

        for ep in range(1, self.config.num_episodes + 1):
            self.episode = ep
            epsilon = self._get_epsilon()

            # Collect experience
            transitions, ep_score = self._play_episode(epsilon)
            self.buffer.push_many(transitions)

            # Train if buffer has enough data
            losses = []
            if len(self.buffer) >= self.config.min_buffer_size:
                for _ in range(self.config.updates_per_episode):
                    loss = self._train_batch()
                    losses.append(loss)

            avg_loss = float(np.mean(losses)) if losses else float("nan")

            # Progress logging (every 50 episodes)
            if ep % 50 == 0 or ep == 1:
                elapsed = time.time() - start_time
                print(
                    f"[ep {ep:5d}] ε={epsilon:.3f} loss={avg_loss:.4f} "
                    f"buf={len(self.buffer):,d} steps={self.global_step:,d} "
                    f"({elapsed:.0f}s)"
                )

            # Save to opponent pool
            if ep % self.config.pool_save_interval == 0 and len(self.buffer) >= self.config.min_buffer_size:
                self.opponent_pool.add(self.model.state_dict(), ep, ep_score)
                # Update opponent model from pool
                pool_weights = self.opponent_pool.sample()
                if pool_weights is not None:
                    self._opponent_model.load_state_dict(pool_weights)
                    self._opponent_model.eval()

            # Evaluation
            if ep % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                eval_metrics["episode"] = ep
                eval_metrics["epsilon"] = round(epsilon, 4)
                eval_metrics["loss"] = round(avg_loss, 6)
                eval_metrics["buffer_size"] = len(self.buffer)
                eval_metrics["global_step"] = self.global_step
                history.append(eval_metrics)

                marker = ""
                if eval_metrics["avg_score"] < best_eval_score:
                    best_eval_score = eval_metrics["avg_score"]
                    self._save_checkpoint("best")
                    marker = " [BEST]"

                print(
                    f"  EVAL @ ep {ep}: score={eval_metrics['avg_score']:.1f}±{eval_metrics['std_score']:.1f} "
                    f"win_rate={eval_metrics['win_rate']:.1%}{marker}"
                )

            # Periodic checkpoint
            if ep % self.config.checkpoint_interval == 0:
                self._save_checkpoint("latest")

        # Final save
        self._save_checkpoint("final")

        # Save history
        history_path = self.config.output_dir / "self_play_history.json"
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\nTraining complete: {self.config.num_episodes} episodes in {elapsed:.0f}s")
        print(f"Best eval score: {best_eval_score:.2f}")
        print(f"Checkpoints saved to: {self.config.output_dir}")

        return {
            "history": history,
            "best_eval_score": best_eval_score,
            "total_episodes": self.config.num_episodes,
            "total_steps": self.global_step,
            "elapsed_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> SelfPlayConfig:
    parser = argparse.ArgumentParser(description="Self-play DQN training for Golf")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--holes-per-game", type=int, default=9)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=3000)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--min-buffer-size", type=int, default=5000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("data/self_play"))
    parser.add_argument("--pool-save-interval", type=int, default=500)
    parser.add_argument("--pool-max-size", type=int, default=10)
    parser.add_argument("--warmstart-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--opponent-types",
        nargs="+",
        default=["heuristic", "self", "pool"],
        choices=["random", "heuristic", "self", "pool"],
    )

    args = parser.parse_args(argv)
    return SelfPlayConfig(
        num_episodes=args.num_episodes,
        holes_per_game=args.holes_per_game,
        updates_per_episode=args.updates_per_episode,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer_size,
        target_update_interval=args.target_update_interval,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        pool_save_interval=args.pool_save_interval,
        pool_max_size=args.pool_max_size,
        warmstart_checkpoint=args.warmstart_checkpoint,
        seed=args.seed,
        device=args.device,
        opponent_types=args.opponent_types,
    )


def main(argv=None) -> None:
    config = parse_args(argv)
    trainer = SelfPlayTrainer(config)
    result = trainer.train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

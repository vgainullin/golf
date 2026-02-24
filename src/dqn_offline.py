"""Offline DQN training for Golf tensor transition logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .tensor_dataset import TensorTransitionDataset


CARD_VOCAB_SIZE = 53  # 52 cards + unknown placeholder
STATE_SEQUENCE_LENGTH = 8  # 6 player cards + holding + discard
NUM_ACTIONS = 16
STAGE_VOCAB_SIZE = 2
STAGE0_LEGAL = torch.tensor(
    [True, True] + [False] * (NUM_ACTIONS - 2),
    dtype=torch.bool,
)
STAGE1_LEGAL = torch.tensor(
    [False, False] + [True] * (NUM_ACTIONS - 2),
    dtype=torch.bool,
)


def mask_illegal_actions(q_values: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
    """Mask out actions that are illegal for the given stage."""
    if stages.ndim != 1:
        raise ValueError("stages tensor must be 1-D (batch of stage indices).")
    device = q_values.device
    stage0_mask = STAGE0_LEGAL.to(device)
    stage1_mask = STAGE1_LEGAL.to(device)
    mask0 = stage0_mask.unsqueeze(0).expand(q_values.size(0), -1)
    mask1 = stage1_mask.unsqueeze(0).expand(q_values.size(0), -1)
    stage_condition = stages.unsqueeze(1) == 0
    mask = torch.where(stage_condition, mask0, mask1)
    return q_values.masked_fill(~mask, float("-inf"))


@dataclass
class TrainingConfig:
    archive_prefix: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 1e-3
    gamma: float = 0.99
    target_update_interval: int = 750
    val_fraction: float = 0.1
    embedding_dim: int = 128
    hidden_dim: int = 256
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    max_steps: Optional[int] = None
    num_workers: int = 0
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 0.0


class OfflineTransitionDataset(Dataset):
    """Torch dataset wrapping offline transition arrays."""

    def __init__(self, transitions: Dict[str, np.ndarray]):
        self.states = torch.from_numpy(transitions["states"].astype(np.int64))
        self.actions = torch.from_numpy(transitions["actions"].astype(np.int64))
        self.rewards = torch.from_numpy(transitions["rewards"].astype(np.float32))
        self.next_states = torch.from_numpy(transitions["next_states"].astype(np.int64))
        self.dones = torch.from_numpy(transitions["dones"].astype(np.float32))
        self.stages = torch.from_numpy(transitions["action_stage"].astype(np.int64))
        self.next_stages = torch.from_numpy(transitions["next_action_stage"].astype(np.int64))

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int):
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index],
            self.stages[index],
            self.next_stages[index],
        )


class GolfDQN(nn.Module):
    """Simple card-aware DQN for Golf actions."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.embedding = nn.Embedding(CARD_VOCAB_SIZE, embedding_dim)
        self.stage_embedding = nn.Embedding(STAGE_VOCAB_SIZE, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear((STATE_SEQUENCE_LENGTH + 1) * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_tokens: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(state_tokens)  # (batch, STATE_SEQUENCE_LENGTH, emb)
        stage_embed = self.stage_embedding(stages).unsqueeze(1)  # (batch, 1, emb)
        combined = torch.cat([embeds, stage_embed], dim=1)
        flat = combined.view(combined.size(0), -1)
        features = self.encoder(flat)
        return self.head(features)


STATE_SEQUENCE_LENGTH_V2 = 29  # 29 card tokens to embed (excludes deck_remaining scalar)


class GolfDQNv2Shallow(nn.Module):
    """Original 2-layer v2 DQN (no LayerNorm). Compatible with older checkpoints."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.embedding = nn.Embedding(CARD_VOCAB_SIZE, embedding_dim)
        self.stage_embedding = nn.Embedding(STAGE_VOCAB_SIZE, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear((STATE_SEQUENCE_LENGTH_V2 + 1) * embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_tokens: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
        card_tokens = state_tokens[:, :29]
        deck_remaining = state_tokens[:, 29].float() / 27.0
        embeds = self.embedding(card_tokens)
        stage_embed = self.stage_embedding(stages).unsqueeze(1)
        combined = torch.cat([embeds, stage_embed], dim=1)
        flat = combined.view(combined.size(0), -1)
        flat = torch.cat([flat, deck_remaining.unsqueeze(1)], dim=1)
        features = self.encoder(flat)
        return self.head(features)


class GolfDQNv2(nn.Module):
    """Expanded DQN with full table visibility (v2 observation)."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.embedding = nn.Embedding(CARD_VOCAB_SIZE, embedding_dim)
        self.stage_embedding = nn.Embedding(STAGE_VOCAB_SIZE, embedding_dim)
        # 29 card tokens + 1 stage token -> flatten, then append 1 scalar (deck remaining)
        self.encoder = nn.Sequential(
            nn.Linear((STATE_SEQUENCE_LENGTH_V2 + 1) * embedding_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_tokens: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
        # state_tokens: (batch, 30) -- first 29 are card tokens, last is deck_remaining
        card_tokens = state_tokens[:, :29]
        deck_remaining = state_tokens[:, 29].float() / 27.0  # normalize

        embeds = self.embedding(card_tokens)  # (batch, 29, emb)
        stage_embed = self.stage_embedding(stages).unsqueeze(1)  # (batch, 1, emb)
        combined = torch.cat([embeds, stage_embed], dim=1)  # (batch, 30, emb)
        flat = combined.view(combined.size(0), -1)  # (batch, 30*emb)
        flat = torch.cat([flat, deck_remaining.unsqueeze(1)], dim=1)  # (batch, 30*emb + 1)
        features = self.encoder(flat)
        return self.head(features)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_transition_arrays(prefix: Path) -> Dict[str, np.ndarray]:
    dataset = TensorTransitionDataset(prefix)
    arrays = dataset.as_qtransformer_arrays()
    mask = arrays["actions"] >= 0

    def _mask_array(name: str) -> np.ndarray:
        return arrays[name][mask]

    player_cards = _mask_array("player_cards")
    holding = _mask_array("holding_cards")[:, None]
    discard = _mask_array("discard_top")
    states = np.concatenate([player_cards, holding, discard], axis=1)

    next_player_cards = _mask_array("next_player_cards")
    next_holding = _mask_array("next_holding_cards")[:, None]
    next_discard = _mask_array("next_discard_top")
    next_states = np.concatenate([next_player_cards, next_holding, next_discard], axis=1)

    return {
        "states": states.astype(np.int64, copy=False),
        "next_states": next_states.astype(np.int64, copy=False),
        "actions": _mask_array("actions").astype(np.int64, copy=False),
        "rewards": _mask_array("rewards").astype(np.float32, copy=False),
        "dones": _mask_array("dones").astype(np.float32, copy=False),
        "action_stage": _mask_array("action_stage").astype(np.int64, copy=False),
        "next_action_stage": _mask_array("next_action_stage").astype(np.int64, copy=False),
    }


def split_dataset(
    dataset: OfflineTransitionDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    if val_fraction <= 0:
        return dataset, None

    num_samples = len(dataset)
    val_size = max(1, int(num_samples * val_fraction))
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    return train_subset, val_subset


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_pref)


def train_one_epoch(
    model: nn.Module,
    target_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gamma: float,
    grad_clip: float,
    target_update_interval: int,
    global_step: int,
    max_steps: Optional[int],
) -> Tuple[float, int, bool]:
    model.train()
    total_loss = 0.0
    batches = 0
    stopped = False

    for batch in loader:
        states, actions, rewards, next_states, dones, stages, next_stages = batch
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        stages = stages.to(device)
        next_stages = next_stages.to(device)

        q_values = model(states, stages)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = target_model(next_states, next_stages)
            masked_next_q = mask_illegal_actions(next_q_values, next_stages)
            next_q_max = masked_next_q.max(dim=1).values
            next_q_max = torch.where(
                torch.isfinite(next_q_max), next_q_max, torch.zeros_like(next_q_max)
            )
            targets = rewards + gamma * (1.0 - dones) * next_q_max

        loss = F.smooth_l1_loss(q_selected, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
        global_step += 1

        if global_step % target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if max_steps is not None and global_step >= max_steps:
            stopped = True
            break

    average_loss = total_loss / max(batches, 1)
    return average_loss, global_step, stopped


def evaluate(
    model: nn.Module,
    target_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    gamma: float,
) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    target_model.eval()
    total_loss = 0.0
    samples = 0

    with torch.no_grad():
        for states, actions, rewards, next_states, dones, stages, next_stages in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            stages = stages.to(device)
            next_stages = next_stages.to(device)

            q_values = model(states, stages)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            next_q_values = target_model(next_states, next_stages)
            masked_next_q = mask_illegal_actions(next_q_values, next_stages)
            next_q = masked_next_q.max(dim=1).values
            next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
            targets = rewards + gamma * (1.0 - dones) * next_q

            loss = F.smooth_l1_loss(q_selected, targets, reduction="sum")
            total_loss += float(loss.item())
            samples += states.size(0)

    if samples == 0:
        return float("nan")
    return total_loss / samples


def train_offline(config: TrainingConfig) -> Dict[str, object]:
    set_seed(config.seed)

    transitions = build_transition_arrays(config.archive_prefix)
    dataset = OfflineTransitionDataset(transitions)
    train_ds, val_ds = split_dataset(dataset, config.val_fraction, config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )

    device = resolve_device(config.device)
    model = GolfDQN(config.embedding_dim, config.hidden_dim).to(device)
    target_model = GolfDQN(config.embedding_dim, config.hidden_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    global_step = 0
    stopped = False

    # Early stopping state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step, stopped = train_one_epoch(
            model=model,
            target_model=target_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            gamma=config.gamma,
            grad_clip=config.grad_clip,
            target_update_interval=config.target_update_interval,
            global_step=global_step,
            max_steps=config.max_steps,
        )
        val_loss = evaluate(model, target_model, val_loader, device, config.gamma)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "steps": global_step,
            }
        )

        # Early stopping check
        if config.early_stopping_patience > 0 and not np.isnan(val_loss):
            if val_loss < (best_val_loss - config.early_stopping_min_delta):
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} steps={global_step} [NEW BEST]"
                )
            else:
                epochs_without_improvement += 1
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} steps={global_step} "
                    f"[no improvement: {epochs_without_improvement}/{config.early_stopping_patience}]"
                )

                if epochs_without_improvement >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    early_stopped = True
                    stopped = True
        else:
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} steps={global_step}"
            )

        if stopped:
            break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "offline_dqn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_state_dict": target_model.state_dict(),
            "config": asdict(config),
            "history": history,
        },
        checkpoint_path,
    )

    history_path = config.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    return {
        "history": history,
        "checkpoint": str(checkpoint_path),
        "history_path": str(history_path),
        "steps": global_step,
    }


def parse_args(argv: Optional[Tuple[str, ...]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train an offline DQN agent on Golf tensor logs."
    )
    parser.add_argument(
        "--archive-prefix",
        default="tmp/tensor_logs_batch/tensor_transitions_combined",
        help="Path prefix to the tensor transition archive (without extension).",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/offline_dqn",
        help="Directory where checkpoints and logs will be written.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-interval", type=int, default=750)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training if validation loss doesn't improve for N epochs (0=disabled)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum change in validation loss to count as improvement",
    )

    args = parser.parse_args(argv)
    return TrainingConfig(
        archive_prefix=Path(args.archive_prefix),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        val_fraction=args.val_fraction,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def main(argv: Optional[Tuple[str, ...]] = None) -> None:
    config = parse_args(argv)
    result = train_offline(config)
    print("Training finished:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

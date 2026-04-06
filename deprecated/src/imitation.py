"""Imitation learning: train a DQN to mimic the heuristic player.

Generates heuristic rollouts using the vectorized engine, then trains
a GolfDQNv2Shallow model with cross-entropy loss to match the heuristic's
action choices. The resulting model can be used as a warm start for RL.

Usage:
    uv run python -m src.imitation --num-games 10000 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dqn_offline import GolfDQNv2Shallow, NUM_ACTIONS, resolve_device, set_seed
from .vectorized_golf import (
    VectorizedGolfState,
    reset_games,
    get_observation_v2,
    get_valid_action_mask,
    step_stage0,
    step_stage1,
    compute_final_score,
    heuristic_stage0,
    heuristic_stage1,
    eps_greedy_batched,
    NUM_ACTIONS as VEC_NUM_ACTIONS,
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_heuristic_data(
    num_games: int,
    holes: int,
    device: torch.device,
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Play all-heuristic games, collect (obs, stage, action) from all seats.

    Returns dict with keys: obs (N,30 int64), stages (N, int64), actions (N, int64).
    """
    all_obs: List[np.ndarray] = []
    all_stages: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []

    games_remaining = num_games
    while games_remaining > 0:
        N = min(games_remaining, batch_size)
        games_remaining -= N

        for hole in range(1, holes + 1):
            state = reset_games(N, device)
            max_rounds = 30

            for _ in range(max_rounds):
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
                    obs_s0 = get_observation_v2(state, pid)
                    action_s0 = heuristic_stage0(state, pid)

                    # Record from active games
                    active_np = active.cpu().numpy()
                    all_obs.append(obs_s0.cpu().numpy()[active_np])
                    all_stages.append(np.zeros(int(active_np.sum()), dtype=np.int64))
                    all_actions.append(action_s0.cpu().numpy()[active_np])

                    step_stage0(state, action_s0, pid)
                    if state.done.all():
                        break

                    # -- Stage 1 --
                    state.current_stage.fill_(1)
                    obs_s1 = get_observation_v2(state, pid)
                    action_s1 = heuristic_stage1(state, pid)

                    active_np = active.cpu().numpy()
                    all_obs.append(obs_s1.cpu().numpy()[active_np])
                    all_stages.append(np.ones(int(active_np.sum()), dtype=np.int64))
                    all_actions.append(action_s1.cpu().numpy()[active_np])

                    step_stage1(state, action_s1, pid)

                    # End-game detection
                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

    return {
        "obs": np.concatenate(all_obs),
        "stages": np.concatenate(all_stages),
        "actions": np.concatenate(all_actions),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_imitation(
    data: Dict[str, np.ndarray],
    hidden_dim: int = 512,
    embedding_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.2,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[nn.Module, List[Dict]]:
    """Train GolfDQNv2Shallow to imitate heuristic actions via cross-entropy."""
    set_seed(seed)

    obs = torch.from_numpy(data["obs"]).long()
    stages = torch.from_numpy(data["stages"]).long()
    actions = torch.from_numpy(data["actions"]).long()

    n = len(obs)
    perm = torch.randperm(n)
    val_size = int(n * val_fraction)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = TensorDataset(obs[train_idx], stages[train_idx], actions[train_idx])
    val_ds = TensorDataset(obs[val_idx], stages[val_idx], actions[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = GolfDQNv2Shallow(embedding_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = [0, 0]
        train_total = [0, 0]

        for batch_obs, batch_stages, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_stages = batch_stages.to(device)
            batch_actions = batch_actions.to(device)

            logits = model(batch_obs, batch_stages)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_obs)
            preds = logits.argmax(dim=1)
            for s in (0, 1):
                mask = batch_stages == s
                if mask.any():
                    train_correct[s] += (preds[mask] == batch_actions[mask]).sum().item()
                    train_total[s] += mask.sum().item()

        train_loss /= len(train_ds)
        train_acc0 = train_correct[0] / max(1, train_total[0])
        train_acc1 = train_correct[1] / max(1, train_total[1])

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = [0, 0]
        val_total = [0, 0]

        with torch.no_grad():
            for batch_obs, batch_stages, batch_actions in val_loader:
                batch_obs = batch_obs.to(device)
                batch_stages = batch_stages.to(device)
                batch_actions = batch_actions.to(device)

                logits = model(batch_obs, batch_stages)
                loss = F.cross_entropy(logits, batch_actions)

                val_loss += loss.item() * len(batch_obs)
                preds = logits.argmax(dim=1)
                for s in (0, 1):
                    mask = batch_stages == s
                    if mask.any():
                        val_correct[s] += (preds[mask] == batch_actions[mask]).sum().item()
                        val_total[s] += mask.sum().item()

        val_loss /= max(1, len(val_ds))
        val_acc0 = val_correct[0] / max(1, val_total[0])
        val_acc1 = val_correct[1] / max(1, val_total[1])

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc_s0": round(train_acc0, 4),
            "train_acc_s1": round(train_acc1, 4),
            "val_acc_s0": round(val_acc0, 4),
            "val_acc_s1": round(val_acc1, 4),
        }
        history.append(record)
        print(
            f"  epoch {epoch:3d}  "
            f"loss={train_loss:.4f}/{val_loss:.4f}  "
            f"acc_s0={train_acc0:.3f}/{val_acc0:.3f}  "
            f"acc_s1={train_acc1:.3f}/{val_acc1:.3f}"
        )

    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    num_games: int,
    holes: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model in [DQN, R, H, R] config. Returns avg score per hole per seat."""
    model.eval()
    N = num_games
    totals = {i: torch.zeros(N, dtype=torch.float32, device=device) for i in range(4)}
    seat_roles = ["dqn", "random", "heuristic", "random"]

    for hole in range(1, holes + 1):
        state = reset_games(N, device)

        for _ in range(30):
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
                    obs = get_observation_v2(state, pid).to(device)
                    sg = torch.zeros(N, dtype=torch.long, device=device)
                    with torch.no_grad():
                        q = model(obs, sg)
                    s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=device)
                    s0_mask[:, 0] = True
                    s0_mask[:, 1] = state.deck_ptr < 52
                    actions_s0 = eps_greedy_batched(q, 0.0, s0_mask)
                else:  # random
                    s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=device)
                    s0_mask[:, 0] = True
                    s0_mask[:, 1] = state.deck_ptr < 52
                    dummy_q = torch.zeros(N, VEC_NUM_ACTIONS, device=device)
                    actions_s0 = eps_greedy_batched(dummy_q, 1.0, s0_mask)

                step_stage0(state, actions_s0, pid)
                if state.done.all():
                    break

                # -- Stage 1 --
                state.current_stage.fill_(1)

                if role == "heuristic":
                    actions_s1 = heuristic_stage1(state, pid)
                elif role == "dqn":
                    obs1 = get_observation_v2(state, pid).to(device)
                    sg1 = torch.ones(N, dtype=torch.long, device=device)
                    with torch.no_grad():
                        q1 = model(obs1, sg1)
                    mask1 = get_valid_action_mask(state, pid)
                    actions_s1 = eps_greedy_batched(q1, 0.0, mask1)
                else:  # random
                    mask1 = get_valid_action_mask(state, pid)
                    dummy_q1 = torch.zeros(N, VEC_NUM_ACTIONS, device=device)
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
            hole_scores = compute_final_score(state.player_cards[:, sid, :], device)
            totals[sid] += hole_scores

    results = {}
    for sid in range(4):
        avg = totals[sid].mean().item() / holes
        results[seat_roles[sid] + f"_seat{sid}"] = round(avg, 3)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Imitation learning: train DQN to mimic heuristic")
    p.add_argument("--num-games", type=int, default=10000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-games", type=int, default=1000)
    p.add_argument("--output", type=str, default="data/model_imitation.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"Device: {device}")

    # 1. Collect data
    t0 = time.time()
    print(f"\nCollecting heuristic data ({args.num_games} games, {args.holes} holes)...")
    data = collect_heuristic_data(args.num_games, args.holes, device)
    n_samples = len(data["obs"])
    n_s0 = (data["stages"] == 0).sum()
    n_s1 = (data["stages"] == 1).sum()
    print(f"  {n_samples:,} samples ({n_s0:,} stage0, {n_s1:,} stage1) in {time.time()-t0:.1f}s")

    # Print action distribution
    for s in (0, 1):
        mask = data["stages"] == s
        acts = data["actions"][mask]
        unique, counts = np.unique(acts, return_counts=True)
        dist = {int(a): int(c) for a, c in zip(unique, counts)}
        print(f"  Stage {s} action distribution: {dist}")

    # 2. Train
    print(f"\nTraining (hidden={args.hidden_dim}, emb={args.embedding_dim}, lr={args.lr})...")
    model, history = train_imitation(
        data,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    # 3. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "model_variant": "v2s",
        },
        "history": history,
    }, out_path)
    print(f"\nSaved to {out_path}")

    # 4. Evaluate
    print(f"\nEvaluating [DQN, R, H, R] x {args.eval_games} games...")
    results = evaluate_model(model, args.eval_games, args.holes, device)
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Also run heuristic baseline for comparison
    print(f"\nBaseline [H, R, H, R] x {args.eval_games} games...")
    from .tournament import TournamentTrainer, TournamentConfig
    tc = TournamentConfig(device=args.device)
    tt = TournamentTrainer.__new__(TournamentTrainer)
    tt.device = device
    baseline, _ = tt._run_eval_config(
        ["heuristic", "random", "heuristic", "random"],
        model=None, obs_fn=None, num_games=args.eval_games, holes=args.holes,
    )
    for sid, avg in baseline.items():
        print(f"  seat{sid}: {avg:.3f}")


if __name__ == "__main__":
    main()

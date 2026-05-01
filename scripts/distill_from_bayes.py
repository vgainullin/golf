"""Distill Bayes Lookahead policy into DQN via pairwise ranking loss.

AlphaZero-style: BL is the oracle (search), DQN is the network being trained.
Collect BL's per-action expected scores on BL-driven trajectories, then
fine-tune the DQN to match BL's action ordering using a pairwise margin
ranking loss (scale-free, no temperature tuning needed).

After distillation, the checkpoint can be resumed with RL training via
src/tournament.py --resume-from.

Usage:
    uv run python -m scripts.distill_from_bayes \\
        --checkpoint data/exp14_win_bonus/gen_350/gen350_agent4.pt \\
        --games 4000 --holes 9 \\
        --epochs 20 --batch-size 512 --lr 1e-4 \\
        --output data/exp14_win_bonus/distilled.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.bayes_optimal import (
    BayesBeliefTracker,
    _best_placement_score,
    expected_score,
    lookahead_stage0,
    lookahead_stage1,
)
from src.tournament import make_model, get_obs_fn
from src.vectorized_golf import (
    NUM_ACTIONS,
    NUM_RANKS,
    get_observation_v2,
    get_valid_action_mask,
    heuristic_stage0,
    heuristic_stage1,
    reset_games,
    step_stage0,
    step_stage1,
)

# Reuse the per-action score functions from the audit script
from scripts.policy_audit import lookahead_stage0_scores, lookahead_stage1_scores


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_expert_data(
    N: int,
    holes: int,
    device: torch.device,
    obs_fn,
) -> Dict[str, np.ndarray]:
    """Run BL on N games x holes. At every player-0 decision record
    (obs, stage, bl_scores, valid_mask).

    BL drives player 0; opponents use heuristic_stage{0,1}.
    """
    tracker = BayesBeliefTracker(N, device=device)

    obs_list: List[np.ndarray] = []
    stage_list: List[np.ndarray] = []
    bl_scores_list: List[np.ndarray] = []
    valid_list: List[np.ndarray] = []

    for hole in range(holes):
        state = reset_games(N, device=device)
        tracker.reset()
        tracker.observe(state, my_player_id=0)

        for _ in range(40):
            if state.done.all():
                break
            for pid in range(4):
                active = ~state.done
                if not active.any():
                    break

                # Stage 0
                state.current_stage.fill_(0)
                if pid == 0:
                    obs = obs_fn(state, pid)  # (N, obs_dim)
                    s0_action, s0_scores = lookahead_stage0_scores(state, pid, tracker)
                    # stage-0 valid: [take=col0, draw=col1]
                    valid_s0 = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)
                    valid_s0[:, 0] = True
                    valid_s0[:, 1] = True

                    obs_list.append(obs[active].cpu().numpy())
                    stage_list.append(np.zeros(int(active.sum()), dtype=np.int64))
                    # Pad s0_scores (N,2) → (N, NUM_ACTIONS) with inf
                    bl_full = torch.full((N, NUM_ACTIONS), float("inf"), device=device)
                    bl_full[:, :2] = s0_scores
                    bl_scores_list.append(bl_full[active].cpu().numpy())
                    valid_list.append(valid_s0[active].cpu().numpy())

                    step_stage0(state, s0_action, pid)
                else:
                    step_stage0(state, heuristic_stage0(state, pid), pid)

                tracker.observe(state, my_player_id=0)
                if state.done.all():
                    break

                # Stage 1
                state.current_stage.fill_(1)
                if pid == 0:
                    obs = obs_fn(state, pid)
                    s1_action, s1_scores = lookahead_stage1_scores(state, pid, tracker)
                    valid_s1 = get_valid_action_mask(state, pid).to(device)

                    obs_list.append(obs[active].cpu().numpy())
                    stage_list.append(np.ones(int(active.sum()), dtype=np.int64))
                    bl_scores_list.append(s1_scores[active].cpu().numpy())
                    valid_list.append(valid_s1[active].cpu().numpy())

                    step_stage1(state, s1_action, pid)
                else:
                    step_stage1(state, heuristic_stage1(state, pid), pid)

                tracker.observe(state, my_player_id=0)

        print(f"  hole {hole + 1}/{holes} done", flush=True)

    return {
        "obs": np.concatenate(obs_list, axis=0),
        "stages": np.concatenate(stage_list, axis=0),
        "bl_scores": np.concatenate(bl_scores_list, axis=0),
        "valid": np.concatenate(valid_list, axis=0),
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(
    q: Tensor,          # (B, A)  DQN Q-values (higher = better)
    bl: Tensor,         # (B, A)  BL expected scores (lower = better), inf for invalid/N/A
    valid: Tensor,      # (B, A)  bool
    margin: float = 0.1,
) -> Tensor:
    """For each pair (i, j) of valid actions where BL strictly prefers i
    (bl[i] < bl[j]), penalise if DQN disagrees (q[i] <= q[j] - margin).

    Loss = mean over all (sample, ordered-pair) triplets.
    """
    B, A = q.shape
    # (B, A, A): broadcast over all pairs
    q_i = q.unsqueeze(2)                      # (B, A, 1)
    q_j = q.unsqueeze(1)                      # (B, 1, A)
    bl_i = bl.unsqueeze(2)                    # (B, A, 1)
    bl_j = bl.unsqueeze(1)                    # (B, 1, A)
    v_i = valid.unsqueeze(2)                  # (B, A, 1)
    v_j = valid.unsqueeze(1)                  # (B, 1, A)

    # BL strictly prefers action i: bl_i < bl_j and both finite
    bl_prefers_i = (bl_i < bl_j) & torch.isfinite(bl_i) & torch.isfinite(bl_j)
    both_valid = v_i & v_j
    pairs = bl_prefers_i & both_valid         # (B, A, A)

    # DQN should have q_i > q_j by at least margin
    loss_ij = F.relu(margin - (q_i - q_j))   # (B, A, A)

    n_pairs = pairs.sum()
    if n_pairs == 0:
        return q.sum() * 0.0                  # keeps graph alive
    return (loss_ij * pairs).sum() / n_pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def distill(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    val_frac: float = 0.1,
) -> None:
    n = len(data["obs"])
    n_val = max(1, int(n * val_frac))
    idx = np.random.permutation(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def to_tensors(subset_idx):
        return (
            torch.from_numpy(data["obs"][subset_idx]).long().to(device),
            torch.from_numpy(data["stages"][subset_idx]).long().to(device),
            torch.from_numpy(data["bl_scores"][subset_idx]).float().to(device),
            torch.from_numpy(data["valid"][subset_idx]).bool().to(device),
        )

    val_obs, val_stages, val_bl, val_valid = to_tensors(val_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(train_idx)
    n_batches = math.ceil(n_train / batch_size)

    print(f"\nDistilling: {n_train} train / {n_val} val samples, "
          f"{epochs} epochs, batch={batch_size}, lr={lr:.2e}, margin={margin}")

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_train)
        train_losses = []
        model.train()
        t0 = time.time()

        for b in range(n_batches):
            bi = train_idx[perm[b * batch_size:(b + 1) * batch_size]]
            obs, stages, bl, valid = to_tensors(bi)

            q = model(obs, stages)
            loss = pairwise_ranking_loss(q, bl, valid, margin)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Validation
        model.eval()
        with torch.no_grad():
            q_val = model(val_obs, val_stages)
            val_loss = pairwise_ranking_loss(q_val, val_bl, val_valid, margin)

        # Agreement rate on val set (greedy)
        with torch.no_grad():
            q_masked = q_val.clone()
            q_masked[~val_valid] = -1e9
            dqn_action = q_masked.argmax(dim=1)

            bl_inf = val_bl.clone()
            bl_inf[~val_valid] = float("inf")
            bl_action = bl_inf.argmin(dim=1)

            agree_rate = (dqn_action == bl_action).float().mean().item()

        elapsed = time.time() - t0
        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train={np.mean(train_losses):.4f}  "
              f"val={float(val_loss):.4f}  "
              f"agree={agree_rate:.1%}  "
              f"({elapsed:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_dqn_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyperparams", {})
    variant = hp.get("model_variant", ckpt.get("variant", "v3"))
    hidden_dim = hp.get("hidden_dim", ckpt.get("hidden_dim", 256))
    emb_dim = hp.get("embedding_dim", 64)

    model = make_model(variant, emb_dim, hidden_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    obs_fn = get_obs_fn(variant)
    print(f"Loaded {path} (variant={variant}, hidden={hidden_dim})")
    return model, obs_fn, variant, hidden_dim, emb_dim, ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="DQN checkpoint to distill into")
    p.add_argument("--games", type=int, default=4000, help="Games to collect per iteration")
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.1,
                   help="Pairwise ranking margin (Q-value units)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, required=True,
                   help="Path to save distilled checkpoint")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model, obs_fn, variant, hidden_dim, emb_dim, orig_ckpt = load_dqn_checkpoint(
        args.checkpoint, device
    )

    print(f"\nCollecting {args.games} games x {args.holes} holes of BL expert data...")
    t0 = time.time()
    data = collect_expert_data(args.games, args.holes, device, obs_fn)
    print(f"Collected {len(data['obs']):,} decisions in {time.time() - t0:.1f}s")

    s0_mask = data["stages"] == 0
    s1_mask = data["stages"] == 1
    print(f"  Stage 0: {s0_mask.sum():,}  Stage 1: {s1_mask.sum():,}")

    distill(model, data, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin)

    # Save: preserve original checkpoint structure so tournament.py can resume
    out_ckpt = {
        "model_state_dict": model.state_dict(),
        "hyperparams": orig_ckpt.get("hyperparams", {
            "model_variant": variant,
            "hidden_dim": hidden_dim,
            "embedding_dim": emb_dim,
        }),
        "distilled_from": args.checkpoint,
        "distill_games": args.games,
        "distill_holes": args.holes,
        "distill_epochs": args.epochs,
    }
    # Carry over optimizer/target if present (tournament.py uses them for resume)
    for key in ("optimizer_state_dict", "target_model_state_dict", "global_step"):
        if key in orig_ckpt:
            out_ckpt[key] = orig_ckpt[key]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    print(f"\nSaved distilled checkpoint: {args.output}")


if __name__ == "__main__":
    main()

"""Residual DQN + DQfD training: improve beyond heuristic-imitation baseline.

Uses a frozen imitation-learned base Q-network plus a trainable residual.
Demo transitions from heuristic rollouts anchor learning via margin loss,
while Boltzmann exploration on Q_total discovers improvements.

Usage:
    uv run python -m src.residual_dqn \
        --base-checkpoint data/model_imitation.pt \
        --num-iterations 100 --output data/model_residual.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dqn_offline import (
    GolfDQNv2Shallow,
    ResidualDQN,
    NUM_ACTIONS,
    mask_illegal_actions,
    resolve_device,
    set_seed,
)
from .imitation import evaluate_model
from .tournament import ReplayBuffer
from .vectorized_golf import (
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
# Demo buffer collection (heuristic rollouts with full transitions)
# ---------------------------------------------------------------------------

def collect_demo_transitions(
    num_games: int,
    holes: int,
    device: torch.device,
    buf: ReplayBuffer,
    batch_size: int = 2048,
) -> None:
    """Play all-heuristic games, record transitions from all 4 seats into buf."""
    games_remaining = num_games
    while games_remaining > 0:
        N = min(games_remaining, batch_size)
        games_remaining -= N

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

                    active_np = active.cpu().numpy()
                    n_active = int(active_np.sum())

                    # Stage 0
                    state.current_stage.fill_(0)
                    obs_s0 = get_observation_v2(state, pid)
                    action_s0 = heuristic_stage0(state, pid)
                    obs_s0_np = obs_s0.cpu().numpy()[active_np]
                    act_s0_np = action_s0.cpu().numpy()[active_np]

                    step_stage0(state, action_s0, pid)
                    if state.done.all():
                        break

                    # Stage 1
                    state.current_stage.fill_(1)
                    obs_s1 = get_observation_v2(state, pid)
                    action_s1 = heuristic_stage1(state, pid)

                    # Record stage 0 transition: obs_s0, act_s0, reward=0, next=obs_s1, done=False
                    obs_s1_np = obs_s1.cpu().numpy()[active_np]
                    buf.push_batch(
                        obs_s0_np, act_s0_np,
                        np.zeros(n_active, dtype=np.float32),
                        obs_s1_np,
                        np.zeros(n_active, dtype=np.float32),
                        np.zeros(n_active, dtype=np.int64),
                        np.ones(n_active, dtype=np.int64),
                    )

                    # Take stage 1 action and get reward
                    reward_s1 = step_stage1(state, action_s1, pid)
                    act_s1_np = action_s1.cpu().numpy()[active_np]
                    rew_s1_np = reward_s1.cpu().numpy()[active_np]

                    # End-game detection
                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

                    # Next obs for stage 1 transition (next player's perspective doesn't matter;
                    # use same player's next obs which will be their next stage 0)
                    # For terminal transitions, next_obs doesn't matter (masked by done)
                    done_np = state.done.cpu().numpy()[active_np].astype(np.float32)
                    # Get next observation for this player
                    state.current_stage.fill_(0)
                    next_obs = get_observation_v2(state, pid).cpu().numpy()[active_np]

                    buf.push_batch(
                        obs_s1_np, act_s1_np,
                        rew_s1_np, next_obs, done_np,
                        np.ones(n_active, dtype=np.int64),
                        np.zeros(n_active, dtype=np.int64),
                    )


# ---------------------------------------------------------------------------
# Agent self-play collection with Boltzmann exploration
# ---------------------------------------------------------------------------

def boltzmann_action(
    q_values: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Select actions using Boltzmann (softmax) exploration over valid actions."""
    masked_q = q_values.clone()
    masked_q[~valid_mask] = float("-inf")
    probs = F.softmax(masked_q / max(temperature, 1e-8), dim=-1)
    return torch.multinomial(probs, 1).squeeze(1)


def collect_agent_transitions(
    model: nn.Module,
    num_games: int,
    holes: int,
    device: torch.device,
    buf: ReplayBuffer,
    temperature: float = 1.0,
    batch_size: int = 2048,
) -> None:
    """Play [DQN, H, H, H] games, record DQN (seat 0) transitions."""
    model.eval()
    seat_roles = ["dqn", "heuristic", "heuristic", "heuristic"]
    games_remaining = num_games
    while games_remaining > 0:
        N = min(games_remaining, batch_size)
        games_remaining -= N

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

                    # Stage 0
                    state.current_stage.fill_(0)
                    if role == "dqn":
                        obs_s0 = get_observation_v2(state, pid)
                        sg0 = torch.zeros(N, dtype=torch.long, device=device)
                        with torch.no_grad():
                            q0 = model(obs_s0, sg0)
                        s0_mask = torch.zeros(N, VEC_NUM_ACTIONS, dtype=torch.bool, device=device)
                        s0_mask[:, 0] = True
                        s0_mask[:, 1] = state.deck_ptr < 52
                        actions_s0 = boltzmann_action(q0, s0_mask, temperature)
                    else:
                        actions_s0 = heuristic_stage0(state, pid)

                    step_stage0(state, actions_s0, pid)
                    if state.done.all():
                        break

                    # Stage 1
                    state.current_stage.fill_(1)
                    if role == "dqn":
                        obs_s1 = get_observation_v2(state, pid)
                        sg1 = torch.ones(N, dtype=torch.long, device=device)
                        with torch.no_grad():
                            q1 = model(obs_s1, sg1)
                        mask1 = get_valid_action_mask(state, pid)
                        actions_s1 = boltzmann_action(q1, mask1, temperature)
                    else:
                        actions_s1 = heuristic_stage1(state, pid)

                    # Record DQN seat transitions
                    if role == "dqn":
                        active_np = active.cpu().numpy()
                        n_active = int(active_np.sum())

                        obs_s0_np = obs_s0.cpu().numpy()[active_np]
                        act_s0_np = actions_s0.cpu().numpy()[active_np]
                        obs_s1_np = obs_s1.cpu().numpy()[active_np]

                        # Stage 0 transition
                        buf.push_batch(
                            obs_s0_np, act_s0_np,
                            np.zeros(n_active, dtype=np.float32),
                            obs_s1_np,
                            np.zeros(n_active, dtype=np.float32),
                            np.zeros(n_active, dtype=np.int64),
                            np.ones(n_active, dtype=np.int64),
                        )

                    reward_s1 = step_stage1(state, actions_s1, pid)

                    # End-game detection
                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

                    # Stage 1 transition for DQN seat
                    if role == "dqn":
                        act_s1_np = actions_s1.cpu().numpy()[active_np]
                        rew_s1_np = reward_s1.cpu().numpy()[active_np]
                        done_np = state.done.cpu().numpy()[active_np].astype(np.float32)
                        state.current_stage.fill_(0)
                        next_obs = get_observation_v2(state, pid).cpu().numpy()[active_np]

                        buf.push_batch(
                            obs_s1_np, act_s1_np,
                            rew_s1_np, next_obs, done_np,
                            np.ones(n_active, dtype=np.int64),
                            np.zeros(n_active, dtype=np.int64),
                        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation from start to end as progress goes 0 -> 1."""
    return start + (end - start) * min(1.0, max(0.0, progress))


def train_residual_dqfd(
    base_checkpoint: Path,
    num_iterations: int = 100,
    games_per_iter: int = 500,
    updates_per_iter: int = 200,
    demo_games: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
    gamma: float = 0.99,
    margin: float = 0.8,
    target_update_interval: int = 500,
    agent_buffer_capacity: int = 200_000,
    demo_buffer_capacity: int = 200_000,
    eval_interval: int = 10,
    eval_games: int = 1000,
    holes: int = 9,
    output: Path = Path("data/model_residual.pt"),
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> List[Dict]:
    set_seed(seed)

    # Load base model
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    embedding_dim = cfg["embedding_dim"]
    hidden_dim = cfg["hidden_dim"]

    base_model = GolfDQNv2Shallow(embedding_dim, hidden_dim).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    # Build residual model (same architecture, fresh weights)
    residual_model = GolfDQNv2Shallow(embedding_dim, hidden_dim).to(device)
    model = ResidualDQN(base_model, residual_model).to(device)

    # Target network tracks residual only (not base logits)
    target_residual = deepcopy(model.residual_model)
    target_residual.eval()

    optimizer = torch.optim.Adam(
        model.residual_model.parameters(), lr=lr,
    )

    # Collect demo buffer
    print(f"Collecting demo transitions ({demo_games} games)...")
    t0 = time.time()
    demo_buf = ReplayBuffer(demo_buffer_capacity)
    collect_demo_transitions(demo_games, holes, device, demo_buf)
    print(f"  {len(demo_buf):,} demo transitions in {time.time()-t0:.1f}s")
    sys.stdout.flush()

    agent_buf = ReplayBuffer(agent_buffer_capacity)
    global_step = 0
    history = []
    best_score = float("inf")

    for iteration in range(1, num_iterations + 1):
        progress = (iteration - 1) / max(1, num_iterations - 1)

        # Schedules
        demo_ratio = linear_schedule(0.25, 0.1, progress)
        lambda_margin = linear_schedule(1.0, 0.1, progress)
        temperature = linear_schedule(1.0, 0.1, progress)

        # Collect agent transitions
        collect_agent_transitions(
            model, games_per_iter, holes, device, agent_buf,
            temperature=temperature,
        )

        if len(agent_buf) < batch_size:
            continue

        # Training updates
        model.train()
        total_dqn_loss = 0.0
        total_margin_loss = 0.0
        n_updates = 0

        n_demo = max(1, int(batch_size * demo_ratio))
        n_agent = batch_size - n_demo

        for _ in range(updates_per_iter):
            # Sample mixed batch
            agent_batch = agent_buf.sample(n_agent, device)
            demo_batch = demo_buf.sample(n_demo, device)

            # Concatenate
            states = torch.cat([agent_batch["states"], demo_batch["states"]])
            actions = torch.cat([agent_batch["actions"], demo_batch["actions"]])
            rewards = torch.cat([agent_batch["rewards"], demo_batch["rewards"]])
            next_states = torch.cat([agent_batch["next_states"], demo_batch["next_states"]])
            dones = torch.cat([agent_batch["dones"], demo_batch["dones"]])
            stages = torch.cat([agent_batch["stages"], demo_batch["stages"]])
            next_stages = torch.cat([agent_batch["next_stages"], demo_batch["next_stages"]])

            # -- DQN loss on Q_residual alone (not Q_base + Q_res) --
            # Q_res learns actual Q-values grounded in rewards
            q_res = model.residual_model(states, stages)
            q_res_selected = q_res.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Action selection uses combined Q (biased toward heuristic)
                next_q_combined = model(next_states, next_stages)
                masked_next_q = mask_illegal_actions(next_q_combined, next_stages)
                next_actions = masked_next_q.argmax(dim=1)

                # Target evaluation uses Q_res only (proper Q-values)
                next_q_res = target_residual(next_states, next_stages)
                next_q_value = next_q_res.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                next_q_value = torch.where(
                    torch.isfinite(next_q_value), next_q_value, torch.zeros_like(next_q_value)
                )
                targets = rewards + gamma * (1.0 - dones) * next_q_value

            dqn_loss = F.smooth_l1_loss(q_res_selected, targets)

            # -- Margin loss on Q_res for demo samples --
            # Q_base already satisfies margin trivially; Q_res needs the signal
            demo_q_res = q_res[n_agent:]  # (n_demo, NUM_ACTIONS)
            demo_actions = actions[n_agent:]
            demo_stages = stages[n_agent:]

            margin_matrix = torch.full_like(demo_q_res, margin)
            margin_matrix.scatter_(1, demo_actions.unsqueeze(1), 0.0)

            demo_q_with_margin = mask_illegal_actions(demo_q_res + margin_matrix, demo_stages)
            max_q_with_margin = demo_q_with_margin.max(dim=1).values
            expert_q = demo_q_res.gather(1, demo_actions.unsqueeze(1)).squeeze(1)
            margin_loss = F.relu(max_q_with_margin - expert_q).mean()

            # Total loss
            loss = dqn_loss + lambda_margin * margin_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.residual_model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            total_dqn_loss += dqn_loss.item()
            total_margin_loss += margin_loss.item()
            n_updates += 1

            # Hard target update
            if global_step % target_update_interval == 0:
                target_residual.load_state_dict(model.residual_model.state_dict())
                target_residual.eval()

        avg_dqn = total_dqn_loss / max(1, n_updates)
        avg_margin = total_margin_loss / max(1, n_updates)

        record = {
            "iteration": iteration,
            "dqn_loss": round(avg_dqn, 5),
            "margin_loss": round(avg_margin, 5),
            "temperature": round(temperature, 3),
            "demo_ratio": round(demo_ratio, 3),
            "lambda_margin": round(lambda_margin, 3),
            "agent_buf_size": len(agent_buf),
            "global_step": global_step,
        }

        # Evaluation
        if iteration % eval_interval == 0 or iteration == 1:
            model.eval()
            results = evaluate_model(model, eval_games, holes, device)
            record["eval"] = results
            dqn_score = results.get("dqn_seat0", float("inf"))

            if dqn_score < best_score:
                best_score = dqn_score
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "residual_state_dict": model.residual_model.state_dict(),
                    "base_config": cfg,
                    "iteration": iteration,
                    "eval": results,
                }, output)
                record["saved"] = True

            print(
                f"[iter {iteration:4d}] dqn={avg_dqn:.4f} margin={avg_margin:.4f} "
                f"temp={temperature:.2f} | "
                f"eval: {results} {'*BEST*' if record.get('saved') else ''}"
            )
        else:
            print(
                f"[iter {iteration:4d}] dqn={avg_dqn:.4f} margin={avg_margin:.4f} "
                f"temp={temperature:.2f}"
            )
        sys.stdout.flush()

        history.append(record)

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Residual DQN + DQfD training")
    p.add_argument("--base-checkpoint", type=str, default="data/model_imitation.pt")
    p.add_argument("--num-iterations", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=500)
    p.add_argument("--updates-per-iter", type=int, default=200)
    p.add_argument("--demo-games", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--margin", type=float, default=0.8)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--agent-buffer-capacity", type=int, default=200_000)
    p.add_argument("--demo-buffer-capacity", type=int, default=200_000)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=1000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--output", type=str, default="data/model_residual.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Iterations: {args.num_iterations}, games/iter: {args.games_per_iter}")
    sys.stdout.flush()

    history = train_residual_dqfd(
        base_checkpoint=Path(args.base_checkpoint),
        num_iterations=args.num_iterations,
        games_per_iter=args.games_per_iter,
        updates_per_iter=args.updates_per_iter,
        demo_games=args.demo_games,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        margin=args.margin,
        target_update_interval=args.target_update_interval,
        agent_buffer_capacity=args.agent_buffer_capacity,
        demo_buffer_capacity=args.demo_buffer_capacity,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        holes=args.holes,
        output=Path(args.output),
        device=device,
        seed=args.seed,
    )

    # Save history
    history_path = Path(args.output).with_suffix(".json")
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}")


if __name__ == "__main__":
    main()

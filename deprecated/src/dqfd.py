"""DQfD (Deep Q-learning from Demonstrations): fine-tune imitation DQN with RL.

Initializes from the imitation checkpoint and trains with mixed demo + self-play
batches. Margin loss prevents catastrophic forgetting, DQN loss learns actual
Q-values, Boltzmann exploration discovers improvements.

Usage:
    uv run python -m src.dqfd \
        --checkpoint data/model_imitation.pt \
        --num-iterations 200 --output data/model_dqfd.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dqn_offline import (
    GolfDQNv2Shallow,
    NUM_ACTIONS,
    mask_illegal_actions,
    resolve_device,
    set_seed,
)
from .imitation import evaluate_model
from .residual_dqn import (
    collect_demo_transitions,
    collect_agent_transitions,
    linear_schedule,
)
from .tournament import ReplayBuffer


def train_dqfd(
    checkpoint: Path,
    num_iterations: int = 200,
    games_per_iter: int = 500,
    updates_per_iter: int = 200,
    demo_games: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
    gamma: float = 0.99,
    margin: float = 0.8,
    target_update_interval: int = 500,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    agent_buffer_capacity: int = 200_000,
    demo_buffer_capacity: int = 200_000,
    eval_interval: int = 10,
    eval_games: int = 1000,
    holes: int = 9,
    output: Path = Path("data/model_dqfd.pt"),
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> List[Dict]:
    set_seed(seed)

    # Load imitation checkpoint as starting point
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    embedding_dim = cfg["embedding_dim"]
    hidden_dim = cfg["hidden_dim"]

    model = GolfDQNv2Shallow(embedding_dim, hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    target_model = deepcopy(model)
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        lambda_margin = 1.0  # keep constant -- decaying caused catastrophic forgetting
        temperature = linear_schedule(temp_start, temp_end, progress)

        # Collect self-play transitions
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
            agent_batch = agent_buf.sample(n_agent, device)
            demo_batch = demo_buf.sample(n_demo, device)

            states = torch.cat([agent_batch["states"], demo_batch["states"]])
            actions = torch.cat([agent_batch["actions"], demo_batch["actions"]])
            rewards = torch.cat([agent_batch["rewards"], demo_batch["rewards"]])
            next_states = torch.cat([agent_batch["next_states"], demo_batch["next_states"]])
            dones = torch.cat([agent_batch["dones"], demo_batch["dones"]])
            stages = torch.cat([agent_batch["stages"], demo_batch["stages"]])
            next_stages = torch.cat([agent_batch["next_stages"], demo_batch["next_stages"]])

            # -- Double DQN loss --
            q_values = model(states, stages)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_online = model(next_states, next_stages)
                masked_next_q = mask_illegal_actions(next_q_online, next_stages)
                next_actions = masked_next_q.argmax(dim=1)

                next_q_target = target_model(next_states, next_stages)
                next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                next_q_value = torch.where(
                    torch.isfinite(next_q_value), next_q_value, torch.zeros_like(next_q_value)
                )
                targets = rewards + gamma * (1.0 - dones) * next_q_value

            dqn_loss = F.smooth_l1_loss(q_selected, targets)

            # -- Margin loss on demo samples --
            demo_q = q_values[n_agent:]
            demo_actions = actions[n_agent:]
            demo_stages = stages[n_agent:]

            margin_matrix = torch.full_like(demo_q, margin)
            margin_matrix.scatter_(1, demo_actions.unsqueeze(1), 0.0)

            demo_q_with_margin = mask_illegal_actions(demo_q + margin_matrix, demo_stages)
            max_q_with_margin = demo_q_with_margin.max(dim=1).values
            expert_q = demo_q.gather(1, demo_actions.unsqueeze(1)).squeeze(1)
            margin_loss = F.relu(max_q_with_margin - expert_q).mean()

            loss = dqn_loss + lambda_margin * margin_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            total_dqn_loss += dqn_loss.item()
            total_margin_loss += margin_loss.item()
            n_updates += 1

            if global_step % target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
                target_model.eval()

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
                    "config": cfg,
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


def main(argv=None):
    p = argparse.ArgumentParser(description="DQfD: fine-tune imitation DQN with RL")
    p.add_argument("--checkpoint", type=str, default="data/model_imitation.pt")
    p.add_argument("--num-iterations", type=int, default=200)
    p.add_argument("--games-per-iter", type=int, default=500)
    p.add_argument("--updates-per-iter", type=int, default=200)
    p.add_argument("--demo-games", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--margin", type=float, default=0.8)
    p.add_argument("--target-update-interval", type=int, default=500)
    p.add_argument("--temp-start", type=float, default=1.0)
    p.add_argument("--temp-end", type=float, default=0.1)
    p.add_argument("--agent-buffer-capacity", type=int, default=200_000)
    p.add_argument("--demo-buffer-capacity", type=int, default=200_000)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=1000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--output", type=str, default="data/model_dqfd.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Iterations: {args.num_iterations}, games/iter: {args.games_per_iter}")
    sys.stdout.flush()

    history = train_dqfd(
        checkpoint=Path(args.checkpoint),
        num_iterations=args.num_iterations,
        games_per_iter=args.games_per_iter,
        updates_per_iter=args.updates_per_iter,
        demo_games=args.demo_games,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        margin=args.margin,
        target_update_interval=args.target_update_interval,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        agent_buffer_capacity=args.agent_buffer_capacity,
        demo_buffer_capacity=args.demo_buffer_capacity,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        holes=args.holes,
        output=Path(args.output),
        device=device,
        seed=args.seed,
    )

    history_path = Path(args.output).with_suffix(".json")
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}")


if __name__ == "__main__":
    main()

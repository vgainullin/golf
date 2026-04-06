"""Compare DQN checkpoints against each other and heuristic baselines.

Usage:
    uv run python -m scripts.eval_compare \
        --checkpoints data/exp11_cyclic/champion.pt data/exp9_v3_extended/champion.pt \
        --games 5000 --holes 9
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.dqn_offline import resolve_device
from src.tournament import make_model, get_obs_fn, TournamentConfig, TournamentTrainer


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    variant = cfg.get("model_variant", "v1")
    hidden_dim = cfg["hidden_dim"]
    embedding_dim = cfg.get("embedding_dim", 128)
    model = make_model(variant, embedding_dim, hidden_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    obs_fn = get_obs_fn(variant)
    agent_id = ckpt.get("agent_record", {}).get("agent_id", Path(path).stem)
    return model, obs_fn, agent_id, variant, hidden_dim


def make_trainer(device):
    config = TournamentConfig(output_dir="/tmp/_eval_compare_dummy", device=str(device))
    trainer = TournamentTrainer.__new__(TournamentTrainer)
    trainer.config = config
    trainer.device = device
    return trainer


def eval_checkpoint(trainer, model, obs_fn, num_games, holes):
    """Evaluate a checkpoint in multiple configs. Returns dict of results."""
    configs = {
        "solo [R,H,R]": ["dqn", "random", "heuristic", "random"],
        "vs 3 random [R,R,R]": ["dqn", "random", "random", "random"],
    }
    results = {}
    for name, seats in configs.items():
        avgs, totals, behavior = trainer._run_eval_config(
            seat_roles=seats,
            model=model,
            obs_fn=obs_fn,
            num_games=num_games,
            holes=holes,
        )
        dqn_scores = (totals[0] / holes).cpu().numpy()
        results[name] = {
            "avg": avgs[0],
            "std": dqn_scores.std(),
            "median": float(np.median(dqn_scores)),
            "behavior": behavior[0] if behavior else {},
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="Paths to .pt checkpoint files")
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    trainer = make_trainer(device)
    G, H = args.games, args.holes

    print(f"Evaluation: {G} games x {H} holes, device={device}\n")

    all_results = []

    for path in args.checkpoints:
        model, obs_fn, agent_id, variant, hidden = load_checkpoint(path, device)
        label = f"{Path(path).parent.name}/{agent_id}"
        print(f"--- {label} (variant={variant}, hidden={hidden}) ---")

        results = eval_checkpoint(trainer, model, obs_fn, G, H)
        for config_name, r in results.items():
            beh = r["behavior"]
            col = beh.get("col_matches", 0)
            rev = beh.get("rev_replace", 0)
            print(f"  {config_name:20s}  avg={r['avg']:.3f}  std={r['std']:.3f}  med={r['median']:.3f}  col={col:.2f}  rev={rev:.2f}")

        all_results.append((label, results))
        print()

    # Reference baselines from MEMORY.md
    print("--- Heuristic baselines (from eval_heuristics.py, 5000 games) ---")
    print(f"  {'solo [R,H,R]':20s}  improved_heuristic=10.52   base_heuristic=14.0")
    print(f"  {'vs 3 random [R,R,R]':20s}  improved_heuristic=8.1")
    print()

    # Summary table
    print("=" * 80)
    print(f"{'Agent':40s} {'[R,H,R]':>8} {'[R,R,R]':>8} {'col':>6} {'rev':>6}")
    print("-" * 80)
    for label, results in all_results:
        solo = results["solo [R,H,R]"]
        rand = results["vs 3 random [R,R,R]"]
        col = solo["behavior"].get("col_matches", 0)
        rev = solo["behavior"].get("rev_replace", 0)
        print(f"{label:40s} {solo['avg']:>8.3f} {rand['avg']:>8.3f} {col:>6.2f} {rev:>6.2f}")
    print(f"{'improved heuristic':40s} {'10.52':>8} {'8.10':>8} {'0.70':>6} {'0.33':>6}")
    print(f"{'base heuristic':40s} {'14.00':>8} {'':>8} {'0.53':>6} {'0.00':>6}")
    print("=" * 80)


if __name__ == "__main__":
    main()

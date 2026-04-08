"""Test the DQN champion on stacked decks vs normal decks.

Loads a checkpoint, runs eval in two configurations on both deck regimes,
and prints a comparison table. Uses TournamentTrainer._run_eval_config which
now accepts a stack_low_cards parameter.

Usage:
    uv run python -m scripts.eval_dqn_stacked \
        --checkpoint data/exp11_cyclic/champion.pt \
        --games 5000 --holes 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dqn_offline import resolve_device
from src.tournament import (
    TournamentConfig,
    TournamentTrainer,
    get_obs_fn,
    make_model,
)


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
    return model, obs_fn, variant, hidden_dim


def make_trainer(device):
    config = TournamentConfig(output_dir="/tmp/_eval_dqn_stacked_dummy", device=str(device))
    trainer = TournamentTrainer.__new__(TournamentTrainer)
    trainer.config = config
    trainer.device = device
    return trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    trainer = make_trainer(device)

    print(f"Loading {args.checkpoint}...")
    model, obs_fn, variant, hidden = load_checkpoint(args.checkpoint, device)
    print(f"  variant={variant}, hidden={hidden}")
    print()
    print(f"Running eval: {args.games} games x {args.holes} holes, device={device}")
    print()

    configs = {
        "solo [_,R,H,R]": ["dqn", "random", "heuristic", "random"],
        "vs 3 random [_,R,R,R]": ["dqn", "random", "random", "random"],
    }

    rows = []
    for name, seats in configs.items():
        # Normal deck
        avgs_n, _, beh_n = trainer._run_eval_config(
            seat_roles=seats,
            model=model,
            obs_fn=obs_fn,
            num_games=args.games,
            holes=args.holes,
            stack_low_cards=False,
        )
        # Stacked deck
        avgs_s, _, beh_s = trainer._run_eval_config(
            seat_roles=seats,
            model=model,
            obs_fn=obs_fn,
            num_games=args.games,
            holes=args.holes,
            stack_low_cards=True,
        )
        rows.append((name, avgs_n[0], avgs_s[0], beh_n[0], beh_s[0]))

    print("=" * 100)
    print(
        f"{'config':<24s} {'normal':>10s} {'stacked':>10s} {'delta':>10s}  "
        f"{'col_n':>6s} {'col_s':>6s}  {'rev_n':>6s} {'rev_s':>6s}"
    )
    print("-" * 100)
    for name, score_n, score_s, beh_n, beh_s in rows:
        delta = score_s - score_n
        col_n = beh_n.get("col_matches", 0)
        col_s = beh_s.get("col_matches", 0)
        rev_n = beh_n.get("rev_replace", 0)
        rev_s = beh_s.get("rev_replace", 0)
        print(
            f"{name:<24s} {score_n:>10.3f} {score_s:>10.3f} {delta:>+10.3f}  "
            f"{col_n:>6.2f} {col_s:>6.2f}  {rev_n:>6.2f} {rev_s:>6.2f}"
        )
    print("=" * 100)
    print()
    print("Reference (from previous experiments, 4p9h, IH on normal/stacked):")
    print("  IH normal [R,H,R]:  ~10.52    IH stacked [R,H,R]:  ~9.15")
    print("  IH normal [R,R,R]:  ~8.10     IH stacked [R,R,R]:  ~9.15")
    print()
    print("delta = stacked - normal. Negative = the stacked deck regime is easier")
    print("(everyone scores fewer points because more low cards are reachable).")


if __name__ == "__main__":
    main()

"""Download hall_of_fame.pt from HF and evaluate [DQN, R, H, R] vs heuristic.

Usage:
    uv run python -m scripts.eval_hof --repo-id vgainullin/golf --games 1000 --holes 9
"""
import argparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from src.dqn_offline import GolfDQNv2, resolve_device
from src.tournament import make_model, get_obs_fn, TournamentConfig, TournamentTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="vgainullin/golf")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)

    print(f"Downloading hall_of_fame.pt from {args.repo_id}...")
    path = hf_hub_download(args.repo_id, "hall_of_fame.pt", repo_type="dataset")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    cfg = ckpt["config"]
    variant = cfg.get("model_variant", "v1")
    hidden_dim = cfg["hidden_dim"]
    embedding_dim = cfg.get("embedding_dim", 128)
    agent_record = ckpt.get("agent_record", {})

    print(f"Loaded: {agent_record.get('agent_id', '?')}  variant={variant}  hidden={hidden_dim}")
    print(f"Checkpoint avg_score: {agent_record.get('hyperparams', {}).get('avg_score', '?')}")
    print()

    model = make_model(variant, embedding_dim, hidden_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_fn = get_obs_fn(variant)

    # Use TournamentTrainer._run_eval_config with a minimal dummy config
    config = TournamentConfig(output_dir="/tmp/_hof_eval_dummy", device=args.device)
    trainer = TournamentTrainer.__new__(TournamentTrainer)
    trainer.config = config
    trainer.device = device

    print(f"Running {args.games} games x {args.holes} holes  [DQN, Random, Heuristic, Random]")
    avgs, totals = trainer._run_eval_config(
        seat_roles=["dqn", "random", "heuristic", "random"],
        model=model,
        obs_fn=obs_fn,
        num_games=args.games,
        holes=args.holes,
    )

    labels = ["DQN (HoF)", "Random", "Heuristic", "Random"]
    print()
    print(f"  {'seat':<12}  {'avg/hole':>9}  {'std':>6}  {'median':>8}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*6}  {'-'*8}")
    for sid in range(4):
        s = (totals[sid] / args.holes).cpu().numpy()
        print(f"  {labels[sid]:<12}  {avgs[sid]:>9.3f}  {s.std():>6.3f}  {np.median(s):>8.3f}")


if __name__ == "__main__":
    main()

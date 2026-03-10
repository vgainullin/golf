"""Analyze card embedding structure from saved checkpoints.

Usage:
    uv run python -m src.analyze_embeddings data/model_imitation.pt
    uv run python -m src.analyze_embeddings data/model_imitation.pt data/tournament_hindsight/gen_020/gen19_agent9.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .vectorized_golf import NUM_RANKS, NUM_SUITS, RANK_SCORES

NUM_CARDS = NUM_RANKS * NUM_SUITS
RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SCORE_VALUES = RANK_SCORES.numpy()


def load_embeddings(path: Path) -> np.ndarray:
    """Load card embedding weights from a checkpoint. Returns (52, emb_dim)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    emb = state["embedding.weight"][:NUM_CARDS].detach().numpy()
    return emb


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / (norms + 1e-8)
    return normed @ normed.T


def rank_cards(rank: int) -> list[int]:
    """Card indices for a given rank (0-12) across all 4 suits."""
    return [suit * NUM_RANKS + rank for suit in range(NUM_SUITS)]


def pairwise_values(matrix: np.ndarray, indices: list[int]) -> list[float]:
    """All pairwise values from a symmetric matrix for given indices."""
    vals = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            vals.append(matrix[indices[i], indices[j]])
    return vals


def analyze(emb: np.ndarray, label: str) -> dict:
    """Print embedding analysis and return summary stats."""
    print(f"\n{'=' * 60}")
    print(f"  {label}  (shape {emb.shape})")
    print(f"{'=' * 60}")

    cos = cosine_sim_matrix(emb)

    # Within-rank vs between-rank cosine similarity
    within, between = [], []
    for rank in range(NUM_RANKS):
        within.extend(pairwise_values(cos, rank_cards(rank)))
    for i in range(NUM_CARDS):
        for j in range(i + 1, NUM_CARDS):
            if i % NUM_RANKS != j % NUM_RANKS:
                between.append(cos[i, j])

    within = np.array(within)
    between = np.array(between)

    print(f"\nCosine similarity:")
    print(f"  Within-rank:  mean={within.mean():.4f}  std={within.std():.4f}  [{within.min():.3f}, {within.max():.3f}]")
    print(f"  Between-rank: mean={between.mean():.4f}  std={between.std():.4f}  [{between.min():.3f}, {between.max():.3f}]")
    sep = within.mean() - between.mean()
    print(f"  Separation:   {sep:+.4f}")

    # L2 distances
    within_l2, between_l2 = [], []
    for rank in range(NUM_RANKS):
        cards = rank_cards(rank)
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                within_l2.append(np.linalg.norm(emb[cards[i]] - emb[cards[j]]))
    for i in range(NUM_CARDS):
        for j in range(i + 1, NUM_CARDS):
            if i % NUM_RANKS != j % NUM_RANKS:
                between_l2.append(np.linalg.norm(emb[i] - emb[j]))
    within_l2 = np.array(within_l2)
    between_l2 = np.array(between_l2)
    print(f"\nL2 distance:")
    print(f"  Within-rank:  mean={within_l2.mean():.4f}  std={within_l2.std():.4f}")
    print(f"  Between-rank: mean={between_l2.mean():.4f}  std={between_l2.std():.4f}")

    # Per-rank breakdown
    print(f"\nPer-rank (cosine sim between suits):")
    for rank in range(NUM_RANKS):
        sims = pairwise_values(cos, rank_cards(rank))
        print(f"  {RANK_NAMES[rank]:>2} (score={SCORE_VALUES[rank]:>3.0f}): mean={np.mean(sims):.4f}  [{min(sims):.3f}, {max(sims):.3f}]")

    # Score-group clustering (cards with same score but different rank)
    score_groups: dict[int, list[int]] = {}
    for rank in range(NUM_RANKS):
        s = int(SCORE_VALUES[rank])
        score_groups.setdefault(s, []).extend(rank_cards(rank))

    multi_rank_groups = {s: cards for s, cards in score_groups.items()
                         if len(set(c % NUM_RANKS for c in cards)) > 1}
    if multi_rank_groups:
        print(f"\nScore-group clustering (same score, different rank):")
        for score in sorted(multi_rank_groups):
            cards = multi_rank_groups[score]
            ranks_in = sorted(set(c % NUM_RANKS for c in cards))
            wr = [cos[cards[i], cards[j]]
                  for i in range(len(cards)) for j in range(i + 1, len(cards))
                  if cards[i] % NUM_RANKS == cards[j] % NUM_RANKS]
            cr = [cos[cards[i], cards[j]]
                  for i in range(len(cards)) for j in range(i + 1, len(cards))
                  if cards[i] % NUM_RANKS != cards[j] % NUM_RANKS]
            rank_str = ",".join(RANK_NAMES[r] for r in ranks_in)
            print(f"  score={score:>3} ({rank_str}): within_rank={np.mean(wr):.4f}  cross_rank={np.mean(cr):.4f}")

    return {"within_cos": within.mean(), "between_cos": between.mean(), "separation": sep}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze card embedding structure.")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="One or more .pt checkpoint paths")
    args = parser.parse_args()

    results = []
    for path in args.checkpoints:
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            continue
        emb = load_embeddings(path)
        stats = analyze(emb, str(path))
        results.append((path, stats))

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"  Summary")
        print(f"{'=' * 60}")
        for path, stats in results:
            print(f"  {path.name:>40s}:  separation={stats['separation']:+.4f}  within={stats['within_cos']:.4f}  between={stats['between_cos']:.4f}")


if __name__ == "__main__":
    main()

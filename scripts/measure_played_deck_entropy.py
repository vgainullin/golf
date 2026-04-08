"""Measure the residual structure of a 'played + casually shuffled' Golf deck.

The scenario being modeled (per user request):

  1. A hand of 4-player 9-hole Golf is played to completion with the
     improved heuristic.
  2. Because IH preferentially places high-value cards (2/K/A and matched
     columns) and discards disasters (10/J/Q), at end-of-hand:
       - the discard pile is biased toward disaster cards
       - the face-up cards in player layouts are biased toward high-value
         cards and matched ranks
  3. A player collects: scoops the discard pile into one hand, sweeps the
     face-up cards into another, stacks them, and gives the result a
     SINGLE riffle shuffle.
  4. The resulting deck has predictable cluster structure -- not uniform.

We model this directly:

  - For N played hands, build the 'collected' deck in this exact order:
      [discard pile] + [face-up layout cards] + [face-down layout cards] + [deck residue]
  - Apply K riffle shuffles for K in {0, 1, 2, 3, 7}.
  - For each position 0..51 in the resulting deck, compute the per-rank
    distribution across the N hands. Aggregate into 3 groups (high-value,
    mid, disaster) for plotting.
  - Plot 5 subplots (one per K) showing P(group | position) across the deck.

What we expect to see:

  K=0: step functions. The first ~8-12 positions (discard pile) are heavily
       biased toward disaster cards. The next ~24 positions (layouts) are
       biased toward high-value cards. The tail (face-down + deck residue)
       is closer to uniform.

  K=1: still strongly clustered. The riffle interleaves the two stacks but
       preserves their bulk identity (positions near 0 and 26 are still
       enriched in their original groups).

  K=2..3: lines flatten progressively but visible residual structure.

  K=7: lines should be nearly flat at the uniform baselines (12/52, 28/52,
       12/52). This is the 'fully randomized' regime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.vectorized_golf import (
    NUM_CARDS,
    NUM_RANKS,
    heuristic_stage0,
    improved_stage1,
    reset_games,
    riffle_shuffle,
    step_stage0,
    step_stage1,
)


DEVICE = torch.device("cpu")

HIGH_VALUE_RANKS = [0, 11, 12]   # 2, K, A
DISASTER_RANKS = [8, 9, 10]      # 10, J, Q
MID_RANKS = [r for r in range(NUM_RANKS) if r not in HIGH_VALUE_RANKS and r not in DISASTER_RANKS]


# ---------------------------------------------------------------------------
# Play hands to completion
# ---------------------------------------------------------------------------


def play_hands_to_completion(num_games: int, n_players: int = 4):
    """Play num_games hands to end-of-hole. Returns the final state."""
    state = reset_games(num_games, DEVICE, n_players=n_players)
    for _ in range(60):
        if state.done.all():
            break
        for pid in range(n_players):
            if state.done.all():
                break
            active = ~state.done
            back_to_trigger = state.last_turn & (state.end_game_player == pid)
            state.done = state.done | (back_to_trigger & active)
            active = ~state.done
            if not active.any():
                break

            state.current_stage.fill_(0)
            a0 = heuristic_stage0(state, pid)
            step_stage0(state, a0, pid)
            if state.done.all():
                break

            state.current_stage.fill_(1)
            a1 = improved_stage1(state, pid)
            step_stage1(state, a1, pid)

            all_rev = state.player_revealed[:, pid, :].all(dim=1)
            newly = active & all_rev & (~state.last_turn)
            state.last_turn = state.last_turn | newly
            state.end_game_player = torch.where(
                newly,
                torch.full_like(state.end_game_player, pid),
                state.end_game_player,
            )
    return state


# ---------------------------------------------------------------------------
# Build collected deck per game
# ---------------------------------------------------------------------------


def collect_played_deck(state, n_players: int = 4, rng=None) -> torch.Tensor:
    """For each game, build a (52,) ordered list of card indices in the
    user's specified collection order:

      1. Discard pile: current top first, then buried cards (in random order
         since the simulator only tracks the buried set, not the order in
         which they were placed -- random within the block is the right
         neutral assumption).
      2. Face-up layout cards (random order within the block).
      3. Face-down layout cards (random order within the block).
      4. Deck residue (in deck buffer order, since the deck IS sequenced).

    Randomizing within each block matters: a deterministic by-card-index
    order creates artifacts because high-value cards (low indices for 2s,
    high indices for K/A) are unevenly represented in each block, so a
    sorted ordering correlates rank with within-block position even though
    the player has no such ordering.

    Returns: (N, 52) int16 tensor.
    """
    if rng is None:
        import random as _random
        rng = _random.Random(0)

    N = state.player_cards.shape[0]
    out = torch.zeros(N, NUM_CARDS, dtype=torch.int16)

    for n in range(N):
        # 1. Discard pile: top + buried in random order
        top = int(state.discard_top[n].item())
        buried = state.discard_buried[n].nonzero().flatten().tolist()
        rng.shuffle(buried)
        discard_block = [top] + buried

        # 2. Face-up layout cards in random order
        face_up = []
        for p in range(n_players):
            for s in range(6):
                if state.player_revealed[n, p, s].item():
                    face_up.append(int(state.player_cards[n, p, s].item()))
        rng.shuffle(face_up)

        # 3. Face-down layout cards in random order
        face_down = []
        for p in range(n_players):
            for s in range(6):
                if not state.player_revealed[n, p, s].item():
                    face_down.append(int(state.player_cards[n, p, s].item()))
        rng.shuffle(face_down)

        # 4. Deck residue in deck buffer order (sequencing IS real here)
        residue = []
        ptr = int(state.deck_ptr[n].item())
        size = int(state.deck_size[n].item())
        for i in range(ptr, size):
            residue.append(int(state.deck[n, i].item()))

        cards = discard_block + face_up + face_down + residue
        if len(cards) != NUM_CARDS:
            raise RuntimeError(
                f"game {n}: collected {len(cards)} cards, expected {NUM_CARDS}. "
                f"This is unexpected -- a hand may have left the simulator in a "
                f"degenerate state."
            )
        out[n] = torch.tensor(cards, dtype=torch.int16)

    return out


# ---------------------------------------------------------------------------
# Per-position group probabilities
# ---------------------------------------------------------------------------


def per_position_group_probs(deck: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given (N, 52) deck of card indices, compute per-position fraction of
    cards in each group. Returns (p_high_value, p_mid, p_disaster), each shape (52,)."""
    ranks = (deck.long() % NUM_RANKS)  # (N, 52)
    hv_mask = (ranks == 0) | (ranks == 11) | (ranks == 12)
    dis_mask = (ranks == 8) | (ranks == 9) | (ranks == 10)
    mid_mask = ~hv_mask & ~dis_mask
    p_hv = hv_mask.float().mean(dim=0).numpy()
    p_mid = mid_mask.float().mean(dim=0).numpy()
    p_dis = dis_mask.float().mean(dim=0).numpy()
    return p_hv, p_mid, p_dis


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_per_position(per_k: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      out_path: Path):
    """Plot per-position group probabilities for each K riffle count."""
    ks = sorted(per_k.keys())
    n_plots = len(ks)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        p_hv, p_mid, p_dis = per_k[k]
        positions = np.arange(NUM_CARDS)
        ax.plot(positions, p_hv, color='tab:green', linewidth=2,
                label='high-value (2/K/A)')
        ax.plot(positions, p_mid, color='tab:gray', linewidth=2,
                label='mid (3..9)')
        ax.plot(positions, p_dis, color='tab:red', linewidth=2,
                label='disaster (10/J/Q)')
        # Uniform baselines
        ax.axhline(12 / 52, color='tab:green', linestyle=':', alpha=0.4)
        ax.axhline(28 / 52, color='tab:gray', linestyle=':', alpha=0.4)
        ax.axhline(12 / 52, color='tab:red', linestyle=':', alpha=0.4)
        title = f"K = {k} riffle{'s' if k != 1 else ''}"
        if k == 0:
            title += " (raw collection)"
        ax.set_title(title)
        ax.set_xlabel("deck position")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("P(card at position is in group)")
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(
        "Played-Golf-deck residual structure: P(group | deck position) after K riffles\n"
        "Collection order: discard pile -> face-up layouts -> face-down layouts -> deck residue",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_entropy_summary(per_k: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                         out_path: Path):
    """Plot avg per-position TV distance from uniform vs K. Single line."""
    ks = sorted(per_k.keys())
    uniform = np.array([12 / 52, 28 / 52, 12 / 52])
    avg_tv = []
    for k in ks:
        p_hv, p_mid, p_dis = per_k[k]
        # TV at each position: 0.5 * sum |p_g - uniform_g|
        per_pos = np.stack([p_hv, p_mid, p_dis], axis=1)  # (52, 3)
        tv_per_pos = 0.5 * np.abs(per_pos - uniform).sum(axis=1)
        avg_tv.append(tv_per_pos.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, avg_tv, marker='o', linewidth=2, color='black')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, label='uniform (random deck)')
    ax.set_xlabel("number of riffle shuffles after collection")
    ax.set_ylabel("avg total-variation distance from uniform\n(avg over 52 positions)")
    ax.set_title("How much residual structure remains after K riffles?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--n-players", type=int, default=4)
    p.add_argument(
        "--ks",
        type=str,
        default="0,1,2,3,7",
        help="comma-separated list of K values (riffle counts) to test",
    )
    p.add_argument("--out-dir", type=str, default="data/figures/belief")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(k) for k in args.ks.split(",")]

    print(f"Playing {args.games} hands of {args.n_players}-player IH golf...")
    state = play_hands_to_completion(args.games, n_players=args.n_players)
    print(f"  done. {int(state.done.sum().item())}/{args.games} hands completed.")

    print("Building collected decks...")
    collected = collect_played_deck(state, n_players=args.n_players)
    print(f"  collected (N, 52) = {tuple(collected.shape)}")

    per_k: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in ks:
        print(f"Applying K={k} riffles...")
        shuffled = riffle_shuffle(collected, k)
        probs = per_position_group_probs(shuffled)
        per_k[k] = probs

    # Sanity print: at K=0, the discard-pile prefix should be very disaster-heavy
    print("\nSanity check (K=0): per-position group prob, first 15 positions")
    p_hv, p_mid, p_dis = per_k[0]
    print(f"  {'pos':>4s} {'P(hv)':>8s} {'P(mid)':>8s} {'P(dis)':>8s}")
    for i in range(15):
        print(f"  {i:>4d} {p_hv[i]:>8.3f} {p_mid[i]:>8.3f} {p_dis[i]:>8.3f}")

    plot_per_position(per_k, out_dir / "played_deck_entropy.png")
    plot_entropy_summary(per_k, out_dir / "played_deck_tv_distance.png")
    print("\nDone.")


if __name__ == "__main__":
    main()

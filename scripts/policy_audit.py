"""Policy audit: compare DQN and Bayes Lookahead decision-by-decision.

Measures:
  1. Agreement rate by stage (stage 0: take/draw; stage 1: placement)
  2. Action ranking correlation (Spearman) -- does DQN order actions like Lookahead?
  3. State context at disagreement points (revealed fraction, hole number, stage)
  4. Counterfactual outcome: same N games run twice (same seed) -- once Lookahead-
     driven, once DQN-driven. Per-hole scores are compared on holes where the agents
     disagree vs holes where they agree throughout.

Usage:
    uv run python -m scripts.policy_audit \\
        --dqn-checkpoint data/exp14_win_bonus/gen_350/gen350_agent4.pt \\
        --games 2000 --holes 9 --seed 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from src.bayes_optimal import (
    BayesBeliefTracker,
    _best_placement_score,
    expected_score,
    lookahead_stage0,
    lookahead_stage1,
)
from src.vectorized_golf import (
    NUM_ACTIONS,
    NUM_RANKS,
    compute_final_score,
    eps_greedy_batched,
    get_valid_action_mask,
    reset_games,
    step_stage0,
    step_stage1,
)

RANK_SCORES = torch.tensor([-2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 0, 1], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Lookahead per-action scores (expose internals for ranking comparison)
# ---------------------------------------------------------------------------

def lookahead_stage1_scores(
    state, player_id: int, tracker: BayesBeliefTracker
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (action, per_action_scores) for stage 1.

    per_action_scores: (N, NUM_ACTIONS) -- expected score for each action,
    inf for invalid actions.
    """
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    cards = state.player_cards[:, player_id, :].clone()
    revealed = state.player_revealed[:, player_id, :].clone()
    held = state.player_holding[:, player_id]

    multiset = tracker.multiset_by_rank()
    total = tracker.total()

    current_e = expected_score(cards, revealed, multiset, total, device)

    scores = torch.full((N, NUM_ACTIONS), float("inf"), device=device)

    has_unrevealed = (~revealed).any(dim=1)
    unrevealed_idx = torch.where(
        ~revealed,
        torch.arange(6, device=device).unsqueeze(0).expand(N, -1),
        torch.full((N, 6), 99, dtype=torch.long, device=device),
    )
    first_unrevealed = unrevealed_idx.min(dim=1).values.clamp(0, 5)

    # Placement actions (2-7): expected score after placing at each position
    for pos in range(6):
        trial_cards = cards.clone()
        trial_cards[:, pos] = held
        trial_revealed = revealed.clone()
        trial_revealed[:, pos] = True
        pos_score = expected_score(trial_cards, trial_revealed, multiset, total, device)
        scores[:, 2 + pos] = pos_score

    # Discard+flip actions (9-14): expected score = current_e (no change in mean)
    for pos in range(6):
        can_flip = ~revealed[:, pos]
        flip_score = torch.where(can_flip, current_e, torch.full_like(current_e, float("inf")))
        scores[:, 9 + pos] = flip_score

    # Get valid mask and zero out invalid
    valid = get_valid_action_mask(state, player_id).to(device)
    scores = torch.where(valid, scores, torch.full_like(scores, float("inf")))

    # Best action = argmin score
    action = scores.argmin(dim=1)
    return action, scores


def lookahead_stage0_scores(
    state, player_id: int, tracker: BayesBeliefTracker
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (action, per_action_scores) for stage 0.

    per_action_scores: (N, 2) -- [e_take, e_draw].
    """
    device = state.player_cards.device
    N = state.player_cards.shape[0]

    cards = state.player_cards[:, player_id, :].clone()
    revealed = state.player_revealed[:, player_id, :].clone()

    multiset = tracker.multiset_by_rank()
    total = tracker.total()
    total_f = total.float().clamp(min=1)
    current_e = expected_score(cards, revealed, multiset, total, device)

    face_card = state.discard_top.long()
    best_take_score, _ = _best_placement_score(cards, revealed, face_card, multiset, total, device)
    e_take = torch.min(best_take_score, current_e)

    e_draw = torch.zeros(N, dtype=torch.float32, device=device)
    for r in range(NUM_RANKS):
        count_r = multiset[:, r].float()
        p_r = count_r / total_f
        if (count_r == 0).all():
            continue
        draw_ms = multiset.clone()
        draw_ms[:, r] = (draw_ms[:, r] - 1).clamp(min=0)
        draw_total = (total - 1).clamp(min=1)
        virtual_held = torch.full((N,), r, dtype=torch.long, device=device)
        best_draw_score, _ = _best_placement_score(cards, revealed, virtual_held, draw_ms, draw_total, device)
        current_e_r = expected_score(cards, revealed, draw_ms, draw_total, device)
        e_draw_r = torch.min(best_draw_score, current_e_r)
        e_draw = e_draw + p_r * e_draw_r

    # scores[:, 0] = e_take, scores[:, 1] = e_draw
    scores = torch.stack([e_take, e_draw], dim=1)
    action = (e_draw < e_take).long()  # 0=take, 1=draw
    return action, scores


# ---------------------------------------------------------------------------
# DQN action + Q-values
# ---------------------------------------------------------------------------

def dqn_action_and_qvalues(
    state, player_id: int, model, obs_fn, device, stage: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = state.player_cards.shape[0]
    obs = obs_fn(state, player_id).to(device)
    sg = torch.full((N,), stage, dtype=torch.long, device=device)
    with torch.no_grad():
        q = model(obs, sg)

    if stage == 0:
        mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)
        mask[:, 0] = True
        mask[:, 1] = state.deck_ptr < 52
    else:
        mask = get_valid_action_mask(state, player_id).to(device)

    q_masked = q.clone()
    q_masked[~mask] = -1e9
    action = q_masked.argmax(dim=1)
    return action, q


# ---------------------------------------------------------------------------
# Single-driver game loop (Lookahead or DQN drives; both are always queried)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_audit_games(
    N: int,
    holes: int,
    device: torch.device,
    model,
    obs_fn,
    dqn_device: torch.device,
    driver: str = "lookahead",  # "lookahead" or "dqn"
) -> Dict:
    """Run N games driven by `driver`. At every decision, query both agents.

    Returns dict with per-decision records and per-hole scores.
    """
    player_id = 0  # we audit seat 0 only
    n_players = 4

    records = []       # per-decision records
    hole_scores = []   # (N,) per hole, for the driver

    for hole in range(holes):
        state = reset_games(N, device, n_players=n_players)
        tracker = BayesBeliefTracker(N, device)
        tracker.reset()
        tracker.observe(state, my_player_id=player_id)

        for _ in range(60):
            if state.done.all():
                break

            for pid in range(n_players):
                active = ~state.done
                back_to_trigger = state.last_turn & (state.end_game_player == pid)
                state.done = state.done | (back_to_trigger & active)
                active = ~state.done
                if not active.any():
                    break

                state.current_stage.fill_(0)

                if pid == player_id:
                    # Query both agents
                    l_a0, l_scores0 = lookahead_stage0_scores(state, pid, tracker)
                    d_a0, d_q0 = dqn_action_and_qvalues(state, pid, model, obs_fn, dqn_device, 0)
                    l_a0_cpu = l_a0.cpu()
                    d_a0_cpu = d_a0.cpu()

                    agree0 = (l_a0_cpu == d_a0_cpu)
                    revealed_frac = state.player_revealed[:, pid, :].float().mean(dim=1).cpu()

                    # Spearman on stage-0 scores (only 2 actions: take/draw)
                    # l_scores0: (N,2), d_q0 columns 0 and 1
                    l_s0_np = l_scores0[:, :2].cpu().numpy()  # lower = better
                    d_q0_np = d_q0[:, :2].cpu().numpy()       # higher = better

                    records.append({
                        "stage": 0,
                        "hole": hole,
                        "agree": agree0.numpy(),
                        "revealed_frac": revealed_frac.numpy(),
                        "l_action": l_a0_cpu.numpy(),
                        "d_action": d_a0_cpu.numpy(),
                        # store raw scores for ranking correlation
                        "l_scores": l_s0_np,   # (N, 2)
                        "d_q": d_q0_np,        # (N, 2)
                    })

                    driver_a0 = l_a0 if driver == "lookahead" else d_a0
                else:
                    driver_a0 = lookahead_stage0(state, pid, tracker)

                step_stage0(state, driver_a0, pid)
                tracker.observe(state, my_player_id=player_id)
                if state.done.all():
                    break

                state.current_stage.fill_(1)

                if pid == player_id:
                    l_a1, l_scores1 = lookahead_stage1_scores(state, pid, tracker)
                    d_a1, d_q1 = dqn_action_and_qvalues(state, pid, model, obs_fn, dqn_device, 1)
                    l_a1_cpu = l_a1.cpu()
                    d_a1_cpu = d_a1.cpu()

                    agree1 = (l_a1_cpu == d_a1_cpu)
                    revealed_frac = state.player_revealed[:, pid, :].float().mean(dim=1).cpu()

                    # For ranking: use valid placement actions (2-7 and 9-14)
                    valid = get_valid_action_mask(state, pid)
                    l_s1_np = l_scores1.cpu().numpy()  # (N, NUM_ACTIONS)
                    d_q1_np = d_q1.cpu().numpy()

                    records.append({
                        "stage": 1,
                        "hole": hole,
                        "agree": agree1.numpy(),
                        "revealed_frac": revealed_frac.numpy(),
                        "l_action": l_a1_cpu.numpy(),
                        "d_action": d_a1_cpu.numpy(),
                        "l_scores": l_s1_np,
                        "d_q": d_q1_np,
                        "valid_mask": valid.numpy(),
                    })

                    driver_a1 = l_a1 if driver == "lookahead" else d_a1
                else:
                    driver_a1 = lookahead_stage1(state, pid, tracker)

                step_stage1(state, driver_a1, pid)
                tracker.observe(state, my_player_id=player_id)

                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last, torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        scores = compute_final_score(state.player_cards[:, player_id, :], device)
        hole_scores.append(scores.cpu().numpy())

    return {"records": records, "hole_scores": hole_scores}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(lookahead_data: Dict, dqn_data: Dict) -> Dict:
    results = {}

    # 1. Agreement rate by stage
    for stage in [0, 1]:
        recs = [r for r in lookahead_data["records"] if r["stage"] == stage]
        all_agree = np.concatenate([r["agree"] for r in recs])
        results[f"agree_rate_s{stage}"] = all_agree.mean()
        results[f"n_decisions_s{stage}"] = len(all_agree)

    # 2. Spearman ranking correlation by stage
    for stage in [0, 1]:
        recs = [r for r in lookahead_data["records"] if r["stage"] == stage]
        corrs = []
        for r in recs:
            N = r["l_scores"].shape[0]
            for i in range(N):
                if stage == 0:
                    l_s = -r["l_scores"][i]   # negate: lower L score = better
                    d_s = r["d_q"][i]
                    if len(l_s) >= 2:
                        c, _ = spearmanr(l_s, d_s)
                        if not np.isnan(c):
                            corrs.append(c)
                else:
                    vm = r.get("valid_mask")
                    if vm is not None:
                        valid = vm[i]
                        l_s = -r["l_scores"][i][valid]  # negate
                        d_s = r["d_q"][i][valid]
                        if len(l_s) >= 2:
                            c, _ = spearmanr(l_s, d_s)
                            if not np.isnan(c):
                                corrs.append(c)
        results[f"spearman_s{stage}"] = np.mean(corrs) if corrs else float("nan")

    # 3. Disagreement context: revealed fraction when they disagree vs agree
    for stage in [0, 1]:
        recs = [r for r in lookahead_data["records"] if r["stage"] == stage]
        agree_rev, disagree_rev = [], []
        for r in recs:
            agree_rev.extend(r["revealed_frac"][r["agree"]].tolist())
            disagree_rev.extend(r["revealed_frac"][~r["agree"]].tolist())
        results[f"revealed_frac_agree_s{stage}"] = np.mean(agree_rev) if agree_rev else float("nan")
        results[f"revealed_frac_disagree_s{stage}"] = np.mean(disagree_rev) if disagree_rev else float("nan")

    # 4. Stage-0 disagreement breakdown: who prefers take vs draw
    recs0 = [r for r in lookahead_data["records"] if r["stage"] == 0]
    l_take, d_take, l_draw, d_draw = 0, 0, 0, 0
    for r in recs0:
        mask = ~r["agree"]
        l_take += (r["l_action"][mask] == 0).sum()
        d_take += (r["d_action"][mask] == 0).sum()
        l_draw += (r["l_action"][mask] == 1).sum()
        d_draw += (r["d_action"][mask] == 1).sum()
    results["disagree_s0_l_take"] = int(l_take)
    results["disagree_s0_d_take"] = int(d_take)
    results["disagree_s0_l_draw"] = int(l_draw)
    results["disagree_s0_d_draw"] = int(d_draw)

    # 5. Counterfactual: compare per-hole scores by whether agents disagreed that hole
    # A hole has a disagreement if any decision in that hole had agree=False
    n_holes = len(lookahead_data["hole_scores"])
    N = lookahead_data["hole_scores"][0].shape[0]

    # Build per-hole disagreement flag
    hole_has_disagree = np.zeros((N, n_holes), dtype=bool)
    for r in lookahead_data["records"]:
        h = r["hole"]
        hole_has_disagree[:, h] |= ~r["agree"]

    l_scores_arr = np.stack(lookahead_data["hole_scores"], axis=1)  # (N, holes)
    d_scores_arr = np.stack(dqn_data["hole_scores"], axis=1)

    agree_mask = ~hole_has_disagree
    disagree_mask = hole_has_disagree

    results["cf_agree_l_score"] = l_scores_arr[agree_mask].mean() if agree_mask.any() else float("nan")
    results["cf_agree_d_score"] = d_scores_arr[agree_mask].mean() if agree_mask.any() else float("nan")
    results["cf_disagree_l_score"] = l_scores_arr[disagree_mask].mean() if disagree_mask.any() else float("nan")
    results["cf_disagree_d_score"] = d_scores_arr[disagree_mask].mean() if disagree_mask.any() else float("nan")
    results["pct_holes_with_disagree"] = hole_has_disagree.mean()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_audit(results: Dict, lookahead_data: Dict, output: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Policy audit: DQN (gen350) vs Bayes Lookahead", fontsize=12, fontweight="bold")

    # Panel 1: agreement rate and ranking correlation by stage
    ax = axes[0, 0]
    stages = ["Stage 0\n(take/draw)", "Stage 1\n(placement)"]
    agree = [results["agree_rate_s0"] * 100, results["agree_rate_s1"] * 100]
    bars = ax.bar(stages, agree, color=["#2471a3", "#e74c3c"], alpha=0.8, width=0.4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    for bar, v in zip(bars, agree):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Agreement rate (%)")
    ax.set_title("Action agreement rate")
    ax.set_ylim(0, 105)

    # Panel 2: Spearman ranking correlation
    ax = axes[0, 1]
    corrs = [results["spearman_s0"], results["spearman_s1"]]
    bars = ax.bar(stages, corrs, color=["#2471a3", "#e74c3c"], alpha=0.8, width=0.4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(1, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
    for bar, v in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Spearman ρ (Q-values vs Lookahead scores)")
    ax.set_title("Action ranking correlation")
    ax.set_ylim(-0.2, 1.1)

    # Panel 3: Stage-0 disagree breakdown (who prefers take vs draw)
    ax = axes[0, 2]
    categories = ["L: take", "D: take", "L: draw", "D: draw"]
    counts = [results["disagree_s0_l_take"], results["disagree_s0_d_take"],
              results["disagree_s0_l_draw"], results["disagree_s0_d_draw"]]
    colors = ["#2471a3", "#e74c3c", "#2471a3", "#e74c3c"]
    alphas = [0.9, 0.9, 0.5, 0.5]
    bars = ax.bar(categories, counts, color=colors, alpha=0.8)
    ax.set_ylabel("Decisions at disagreement points")
    ax.set_title("Stage-0 disagreements:\nwho prefers take vs draw?")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(counts) * 0.01,
                str(v), ha="center", fontsize=9)

    # Panel 4: Revealed fraction at agree vs disagree points
    ax = axes[1, 0]
    x = np.arange(2)
    w = 0.3
    for si, (stage, color) in enumerate([(0, "#2471a3"), (1, "#e74c3c")]):
        agree_rv = results[f"revealed_frac_agree_s{stage}"]
        disagree_rv = results[f"revealed_frac_disagree_s{stage}"]
        ax.bar(x + si * w, [agree_rv, disagree_rv], w, color=color, alpha=0.8,
               label=f"Stage {stage}")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["Agree", "Disagree"])
    ax.set_ylabel("Avg fraction of cards revealed")
    ax.set_title("Board state at decision point\n(agree vs disagree)")
    ax.legend(fontsize=8)

    # Panel 5: Counterfactual hole scores
    ax = axes[1, 1]
    labels = ["Agree holes\n(Lookahead)", "Agree holes\n(DQN)",
              "Disagree holes\n(Lookahead)", "Disagree holes\n(DQN)"]
    values = [results["cf_agree_l_score"], results["cf_agree_d_score"],
              results["cf_disagree_l_score"], results["cf_disagree_d_score"]]
    colors = ["#2471a3", "#e74c3c", "#2471a3", "#e74c3c"]
    bars = ax.bar(labels, values, color=colors)
    for bar, a in zip(bars, [0.9, 0.9, 0.5, 0.5]):
        bar.set_alpha(a)
    ax.set_ylabel("Avg score / hole (lower = better)")
    ax.set_title(f"Counterfactual: hole scores\n({results['pct_holes_with_disagree']:.1%} of holes have ≥1 disagreement)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)

    # Panel 6: Per-hole disagreement rate over hole number
    ax = axes[1, 2]
    n_holes = len(lookahead_data["hole_scores"])
    N = lookahead_data["hole_scores"][0].shape[0]
    hole_has_disagree = np.zeros((N, n_holes), dtype=bool)
    for r in lookahead_data["records"]:
        hole_has_disagree[:, r["hole"]] |= ~r["agree"]
    disagree_rate_per_hole = hole_has_disagree.mean(axis=0)
    ax.bar(np.arange(1, n_holes + 1), disagree_rate_per_hole * 100,
           color="#7f8c8d", alpha=0.8)
    ax.set_xlabel("Hole number")
    ax.set_ylabel("% of games with ≥1 disagreement")
    ax.set_title("Disagreement rate by hole")
    ax.set_xticks(np.arange(1, n_holes + 1))

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dqn-checkpoint", required=True)
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--holes", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="data/figures/policy_audit.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from src.tournament import make_model, get_obs_fn
    ckpt = torch.load(args.dqn_checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    variant = cfg.get("model_variant", "v1")
    hidden_dim = cfg["hidden_dim"]
    embedding_dim = cfg.get("embedding_dim", 128)
    model = make_model(variant, embedding_dim, hidden_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    obs_fn = get_obs_fn(variant)
    print(f"Loaded DQN from {args.dqn_checkpoint} (variant={variant}, hidden={hidden_dim})")

    N, H = args.games, args.holes

    print(f"\nRunning {N} games x {H} holes driven by Lookahead (querying DQN at every step)...")
    torch.manual_seed(args.seed)
    lookahead_data = run_audit_games(N, H, device, model, obs_fn, device, driver="lookahead")

    print(f"Running {N} games x {H} holes driven by DQN (same seed)...")
    torch.manual_seed(args.seed)
    dqn_data = run_audit_games(N, H, device, model, obs_fn, device, driver="dqn")

    print("\nAnalyzing...")
    results = analyze(lookahead_data, dqn_data)

    print(f"\n=== Policy Audit: DQN vs Lookahead ({N} games x {H} holes) ===\n")
    print(f"  Agreement rate:")
    print(f"    Stage 0 (take/draw):  {results['agree_rate_s0']:.1%}  ({results['n_decisions_s0']:,} decisions)")
    print(f"    Stage 1 (placement):  {results['agree_rate_s1']:.1%}  ({results['n_decisions_s1']:,} decisions)")
    print(f"\n  Action ranking correlation (Spearman ρ):")
    print(f"    Stage 0: {results['spearman_s0']:.3f}")
    print(f"    Stage 1: {results['spearman_s1']:.3f}")
    print(f"\n  Stage-0 disagreements (Lookahead vs DQN preference):")
    print(f"    Lookahead prefers take: {results['disagree_s0_l_take']:,}  |  DQN prefers take: {results['disagree_s0_d_take']:,}")
    print(f"    Lookahead prefers draw: {results['disagree_s0_l_draw']:,}  |  DQN prefers draw: {results['disagree_s0_d_draw']:,}")
    print(f"\n  Revealed fraction at decision points:")
    print(f"    Stage 0 -- agree: {results['revealed_frac_agree_s0']:.3f}  disagree: {results['revealed_frac_disagree_s0']:.3f}")
    print(f"    Stage 1 -- agree: {results['revealed_frac_agree_s1']:.3f}  disagree: {results['revealed_frac_disagree_s1']:.3f}")
    print(f"\n  Counterfactual hole scores ({results['pct_holes_with_disagree']:.1%} of holes have ≥1 disagreement):")
    print(f"    Agree holes:    Lookahead={results['cf_agree_l_score']:.3f}  DQN={results['cf_agree_d_score']:.3f}  gap={results['cf_agree_d_score']-results['cf_agree_l_score']:+.3f}")
    print(f"    Disagree holes: Lookahead={results['cf_disagree_l_score']:.3f}  DQN={results['cf_disagree_d_score']:.3f}  gap={results['cf_disagree_d_score']-results['cf_disagree_l_score']:+.3f}")

    plot_audit(results, lookahead_data, args.output)


if __name__ == "__main__":
    main()

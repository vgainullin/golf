"""General RL diagnostics for MDP correctness.

Pre-training probes:
  1. Transition fidelity - does next_obs match actual next observation?
  2. Reward-action distribution - systematic bias in reward by action?
  3. Determinism - same seed + same actions = same trajectory?
  4. Observation sanity - NaN, Inf, shape, purity checks

Passive checks (integrated into training loop):
  - NaN/Inf in rewards and observations
  - Shape consistency
  - Action validity

Usage:
    uv run python -m src.diagnostics
    uv run python -m src.diagnostics --check fidelity
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .vectorized_golf import (
    VectorizedGolfState,
    reset_games,
    get_observation,
    get_observation_v2,
    step_stage0,
    step_stage1,
    get_valid_action_mask,
    heuristic_stage0,
    heuristic_stage1,
    compute_final_score,
    compute_score,
    eps_greedy_batched,
    NUM_ACTIONS,
)
from .reward_shaping import HindsightRewardShaper


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransitionBatch:
    """Raw transition data from random rollouts. Game-agnostic."""
    obs: np.ndarray               # (T, obs_dim)
    actions: np.ndarray           # (T,)
    rewards: np.ndarray           # (T,)
    next_obs: np.ndarray          # (T, obs_dim) - correct (deferred) next observation
    next_obs_immediate: np.ndarray  # (T, obs_dim) - captured right after action
    dones: np.ndarray             # (T,)
    stages: np.ndarray            # (T,) - 0 or 1
    reference_rewards: Optional[np.ndarray] = None  # ground-truth reward if available


@dataclass
class CheckResult:
    name: str
    status: str   # "OK", "FLAG", "FAIL"
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        tag = f"[{self.status}]"
        return f"{self.name}: {tag}\n{self.message}"


# ---------------------------------------------------------------------------
# Pre-training probes (game-agnostic)
# ---------------------------------------------------------------------------

def check_transition_fidelity(batch: TransitionBatch) -> CheckResult:
    """Compare deferred next_obs to immediate next_obs. Report mismatch rate."""
    T = len(batch.obs)
    diff = (batch.next_obs != batch.next_obs_immediate)
    mismatch_per_row = diff.any(axis=1)
    n_mismatch = int(mismatch_per_row.sum())
    pct = n_mismatch / T * 100

    l1 = np.abs(batch.next_obs.astype(np.float64) - batch.next_obs_immediate.astype(np.float64)).sum(axis=1)
    mean_l1 = float(l1.mean())

    lines = [f"  immediate != deferred: {pct:.1f}% ({n_mismatch}/{T})"]
    lines.append(f"  mean L1 distance: {mean_l1:.2f}")

    if mean_l1 > 1.0:
        status = "FLAG"
        lines.append("  > Large gap between immediate and deferred next_obs.")
        lines.append("    Verify replay buffer stores the deferred version.")
    else:
        status = "OK"

    return CheckResult(
        name=f"Transition Fidelity ({T} transitions)",
        status=status,
        message="\n".join(lines),
        details={"mismatch_pct": round(pct, 1), "mean_l1": round(mean_l1, 2), "n": T},
    )


def check_reward_action_distribution(batch: TransitionBatch) -> CheckResult:
    """Group rewards by action for stage-1 transitions. Flag large spread."""
    # Only analyze stage-1 transitions (stage-0 always has reward 0)
    s1_mask = batch.stages == 1
    actions = batch.actions[s1_mask]
    rewards = batch.rewards[s1_mask]
    unique_actions = sorted(np.unique(actions))

    rows = []
    means = []
    for a in unique_actions:
        mask = actions == a
        r = rewards[mask]
        m = float(r.mean())
        s = float(r.std()) if len(r) > 1 else 0.0
        rows.append((a, m, s, int(mask.sum())))
        means.append(m)

    spread = max(means) - min(means) if means else 0.0

    lines = []
    lines.append("  action | mean    | std   | n")
    lines.append("  -------+---------+-------+------")
    for a, m, s, n in rows:
        lines.append(f"  {a:>5d}  | {m:>7.3f} | {s:>5.3f} | {n:>5d}")
    lines.append(f"  spread: {spread:.2f}")

    status = "OK"
    if spread > 2.0:
        status = "FLAG"
        lines.append("  > Large spread. Check if this reflects true action value")
        lines.append("    or a structural reward bias.")

    # Reference rewards
    if batch.reference_rewards is not None:
        ref_rewards = batch.reference_rewards[s1_mask]
        ref_means = []
        for a in unique_actions:
            mask = actions == a
            r = ref_rewards[mask]
            m = float(r.mean())
            ref_means.append(m)
        ref_spread = max(ref_means) - min(ref_means) if ref_means else 0.0
        lines.append(f"  reference reward spread: {ref_spread:.2f}")
        if ref_spread < spread * 0.5:
            lines.append("  > Reference rewards show less bias.")

    T = len(batch.obs)
    return CheckResult(
        name=f"Reward-Action Distribution ({T} transitions)",
        status=status,
        message="\n".join(lines),
        details={"spread": round(spread, 2), "n": T},
    )


def check_determinism(collect_fn: Callable, seed: int = 42) -> CheckResult:
    """Run two episodes with same seed and same actions. Assert trajectories match."""
    batch1 = collect_fn(num_games=50, seed=seed)
    batch2 = collect_fn(num_games=50, seed=seed)

    lines = []
    lines.append(f"  2 collections, seed={seed}, {len(batch1.obs)} steps each")

    obs_match = np.array_equal(batch1.obs, batch2.obs)
    reward_match = np.allclose(batch1.rewards, batch2.rewards, atol=1e-6)
    done_match = np.array_equal(batch1.dones, batch2.dones)
    action_match = np.array_equal(batch1.actions, batch2.actions)

    lines.append(f"  observations: {'identical' if obs_match else 'DIFFER'}")
    lines.append(f"  rewards: {'identical' if reward_match else 'DIFFER'}")
    lines.append(f"  dones: {'identical' if done_match else 'DIFFER'}")
    lines.append(f"  actions: {'identical' if action_match else 'DIFFER'}")

    all_ok = obs_match and reward_match and done_match and action_match
    if not all_ok:
        status = "FAIL"
        if not obs_match:
            diffs = (batch1.obs != batch2.obs).any(axis=1)
            first_diff = int(np.argmax(diffs))
            lines.append(f"  First obs mismatch at step {first_diff}")
    else:
        status = "OK"

    return CheckResult(
        name="Determinism",
        status=status,
        message="\n".join(lines),
        details={"obs_match": obs_match, "reward_match": reward_match,
                 "done_match": done_match, "action_match": action_match},
    )


def check_observation_sanity(batch: TransitionBatch) -> CheckResult:
    """Check obs/next_obs for NaN, Inf, shape consistency, dtype range."""
    all_obs = np.concatenate([batch.obs, batch.next_obs], axis=0)
    T = len(all_obs)

    n_nan = int(np.isnan(all_obs.astype(np.float64)).any(axis=1).sum())
    n_inf = int(np.isinf(all_obs.astype(np.float64)).any(axis=1).sum())
    shape_ok = batch.obs.shape[1:] == batch.next_obs.shape[1:]

    lines = []
    lines.append(f"  {T} observations checked")
    lines.append(f"  NaN: {n_nan}  Inf: {n_inf}  shape match: {'OK' if shape_ok else 'MISMATCH'}")

    failed = n_nan > 0 or n_inf > 0 or not shape_ok
    status = "FAIL" if failed else "OK"

    return CheckResult(
        name="Observation Sanity",
        status=status,
        message="\n".join(lines),
        details={"n_nan": n_nan, "n_inf": n_inf, "shape_ok": shape_ok, "n": T},
    )


# ---------------------------------------------------------------------------
# Passive checks (call per batch during training)
# ---------------------------------------------------------------------------

def assert_transition_batch(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    num_actions: int,
) -> None:
    """Vectorized assertions on a batch of transitions. Raises on first violation."""
    obs_f = obs.astype(np.float64)
    next_f = next_obs.astype(np.float64)

    assert not np.any(np.isnan(obs_f)), "NaN in obs"
    assert not np.any(np.isnan(next_f)), "NaN in next_obs"
    assert not np.any(np.isnan(rewards)), "NaN in rewards"
    assert not np.any(np.isinf(rewards)), "Inf in rewards"
    assert obs.shape[1:] == next_obs.shape[1:], (
        f"obs/next_obs shape mismatch: {obs.shape} vs {next_obs.shape}"
    )
    assert np.all((actions >= 0) & (actions < num_actions)), (
        f"Invalid action(s): min={actions.min()}, max={actions.max()}, num_actions={num_actions}"
    )


# ---------------------------------------------------------------------------
# Game-specific collection (Golf-aware)
# ---------------------------------------------------------------------------

def collect_golf_transitions(
    num_games: int = 500,
    num_holes: int = 1,
    device: str = "cpu",
    include_reference: bool = True,
    seed: Optional[int] = None,
) -> TransitionBatch:
    """Run random-policy games, recording both immediate and deferred next_obs.

    The collector mimics the training loop's structure:
    - Stage 0: record immediately (no deferral needed, obs changes only for acting player)
    - Stage 1: record both immediate next_obs and deferred next_obs (after opponents act)

    Also checks observation purity: calls obs function twice on same state, asserts equal.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dev = torch.device(device)
    N = num_games
    LEARNER = 2  # match training loop seat assignment

    obs_fn = get_observation_v2  # v2 gives the richest observation (30 tokens)
    reward_shaper = HindsightRewardShaper() if include_reference else None

    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_next_obs_imm = []
    all_dones = []
    all_stages = []
    all_ref_rewards = []

    for hole in range(1, num_holes + 1):
        state = reset_games(N, dev)
        max_rounds = 30
        pending_s1 = None  # (obs, act, rew, imm_next, active, ref_rew)

        for round_num in range(max_rounds):
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
                obs = obs_fn(state, pid)

                # Observation purity check: call twice, assert equal
                obs_check = obs_fn(state, pid)
                assert torch.equal(obs, obs_check), "Observation function is not pure!"

                # Complete pending stage-1 transition (deferred next_obs)
                if pid == LEARNER and pending_s1 is not None:
                    p_obs, p_act, p_rew, p_imm, p_active, p_ref = pending_s1
                    next_obs_deferred = obs.cpu().numpy()
                    done_np = state.done.cpu().numpy()

                    for i in range(N):
                        if p_active[i]:
                            all_obs.append(p_obs[i])
                            all_actions.append(p_act[i])
                            all_rewards.append(p_rew[i])
                            all_next_obs.append(next_obs_deferred[i])
                            all_next_obs_imm.append(p_imm[i])
                            all_dones.append(done_np[i])
                            all_stages.append(1)
                            if p_ref is not None:
                                all_ref_rewards.append(p_ref[i])
                    pending_s1 = None

                # Random stage 0 action
                s0_mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=dev)
                s0_mask[:, 0] = True
                s0_mask[:, 1] = state.deck_ptr < 52
                dummy_q = torch.zeros(N, NUM_ACTIONS, device=dev)
                actions_s0 = eps_greedy_batched(dummy_q, 1.0, s0_mask)

                # Record stage-0 transition for learner
                if pid == LEARNER:
                    obs_before = obs.cpu().numpy()
                    aid_s0 = actions_s0.cpu().numpy()

                step_stage0(state, actions_s0, pid)

                if pid == LEARNER:
                    obs_after = obs_fn(state, pid).cpu().numpy()
                    active_np = active.cpu().numpy()
                    done_np = state.done.cpu().numpy()
                    for i in range(N):
                        if active_np[i]:
                            all_obs.append(obs_before[i])
                            all_actions.append(aid_s0[i])
                            all_rewards.append(0.0)  # stage 0 always returns 0
                            all_next_obs.append(obs_after[i])
                            all_next_obs_imm.append(obs_after[i])  # same for stage 0
                            all_dones.append(done_np[i])
                            all_stages.append(0)
                            if include_reference:
                                all_ref_rewards.append(0.0)

                if state.done.all():
                    break

                # -- Stage 1 --
                state.current_stage.fill_(1)

                # Random stage 1 action
                mask1 = get_valid_action_mask(state, pid)
                dummy_q1 = torch.zeros(N, NUM_ACTIONS, device=dev)
                actions_s1 = eps_greedy_batched(dummy_q1, 1.0, mask1)

                # Capture stage-1 data for learner
                if pid == LEARNER:
                    obs_before_s1 = obs_fn(state, pid).cpu().numpy()
                    aid_s1 = actions_s1.cpu().numpy()
                    if reward_shaper is not None:
                        cards_before = state.player_cards[:, pid, :].clone()

                rewards_s1 = step_stage1(state, actions_s1, pid)

                if pid == LEARNER:
                    # Immediate next_obs (right after action, before opponents)
                    imm_next = obs_fn(state, pid).cpu().numpy()
                    raw_rew = rewards_s1.cpu().numpy()

                    ref_rew = None
                    if reward_shaper is not None:
                        cards_after = state.player_cards[:, pid, :].clone()
                        shaped = reward_shaper.shape(
                            rewards_s1, cards_before=cards_before,
                            cards_after=cards_after, active=active, device=dev,
                        )
                        ref_rew = shaped.cpu().numpy()

                    pending_s1 = (
                        obs_before_s1, aid_s1, raw_rew,
                        imm_next, active.cpu().numpy(), ref_rew,
                    )

                # Check last turn
                all_rev = state.player_revealed[:, pid, :].all(dim=1)
                newly_last = active & all_rev & (~state.last_turn)
                state.last_turn = state.last_turn | newly_last
                state.end_game_player = torch.where(
                    newly_last,
                    torch.full_like(state.end_game_player, pid),
                    state.end_game_player,
                )

        # Flush pending stage-1 as terminal
        if pending_s1 is not None:
            p_obs, p_act, p_rew, p_imm, p_active, p_ref = pending_s1
            for i in range(N):
                if p_active[i]:
                    all_obs.append(p_obs[i])
                    all_actions.append(p_act[i])
                    all_rewards.append(p_rew[i])
                    all_next_obs.append(p_obs[i])  # dummy, masked by done=1
                    all_next_obs_imm.append(p_imm[i])
                    all_dones.append(1.0)
                    all_stages.append(1)
                    if p_ref is not None:
                        all_ref_rewards.append(p_ref[i])
            pending_s1 = None

    result = TransitionBatch(
        obs=np.array(all_obs),
        actions=np.array(all_actions, dtype=np.int64),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_obs=np.array(all_next_obs),
        next_obs_immediate=np.array(all_next_obs_imm),
        dones=np.array(all_dones, dtype=np.float32),
        stages=np.array(all_stages, dtype=np.int64),
        reference_rewards=np.array(all_ref_rewards, dtype=np.float32) if all_ref_rewards else None,
    )
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_checks(
    batch: TransitionBatch,
    collect_fn: Optional[Callable] = None,
) -> List[CheckResult]:
    """Run all pre-training diagnostic checks."""
    results = []
    results.append(check_transition_fidelity(batch))
    results.append(check_reward_action_distribution(batch))
    if collect_fn is not None:
        results.append(check_determinism(collect_fn))
    results.append(check_observation_sanity(batch))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="RL MDP diagnostics for Golf")
    p.add_argument("--check", type=str, default=None,
                   choices=["fidelity", "reward", "determinism", "sanity"],
                   help="Run a single check instead of all")
    p.add_argument("--num-games", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    print("=== Pre-training MDP Diagnostics ===\n")

    def collect_fn(num_games=None, seed=None):
        return collect_golf_transitions(
            num_games=num_games or args.num_games,
            device=args.device, seed=seed,
        )

    if args.check == "determinism":
        results = [check_determinism(collect_fn)]
    else:
        print("Collecting transitions...")
        batch = collect_golf_transitions(
            num_games=args.num_games, device=args.device,
        )
        print(f"Collected {len(batch.obs)} transitions\n")

        if args.check == "fidelity":
            results = [check_transition_fidelity(batch)]
        elif args.check == "reward":
            results = [check_reward_action_distribution(batch)]
        elif args.check == "sanity":
            results = [check_observation_sanity(batch)]
        else:
            results = run_all_checks(batch, collect_fn)

    for r in results:
        print(r)
        print()

    ok = sum(1 for r in results if r.status == "OK")
    flag = sum(1 for r in results if r.status == "FLAG")
    fail = sum(1 for r in results if r.status == "FAIL")
    print(f"=== {len(results)} checks complete: {ok} OK, {flag} FLAG, {fail} FAIL ===")

    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""LLM Golf player harness -- play golf via OpenRouter API.

Usage:
    uv run python -m src.llm_player --model anthropic/claude-sonnet-4-5 --games 5 --holes 9
    uv run python -m src.llm_player --model google/gemini-2.5-flash --games 1 --holes 1 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI

from src.vectorized_golf import (
    NUM_ACTIONS,
    NUM_RANKS,
    RANK_SCORES,
    UNKNOWN_CARD,
    VectorizedGolfState,
    compute_final_score,
    compute_score,
    count_column_matches,
    get_valid_action_mask,
    heuristic_stage0,
    heuristic_stage1,
    reset_games,
    step_stage0,
    step_stage1,
)

# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUIT_NAMES = ["spades", "diamonds", "clubs", "hearts"]
SUIT_SYMBOLS = {"spades": "\u2660", "diamonds": "\u2666", "clubs": "\u2663", "hearts": "\u2665"}
SCORE_BY_RANK = [-2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 0, 1]


def card_str(idx: int) -> str:
    """Card index (0-51) to human string like '7\u2660'. 52 or -1 -> '??'."""
    if idx < 0 or idx >= 52:
        return "??"
    rank = idx % NUM_RANKS
    suit = idx // NUM_RANKS
    return f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[SUIT_NAMES[suit]]}"


def card_score_str(idx: int) -> str:
    """Card index to score string, e.g. '7\u2660 (score 7)'."""
    if idx < 0 or idx >= 52:
        return "??"
    rank = idx % NUM_RANKS
    return f"{card_str(idx)} (score {SCORE_BY_RANK[rank]})"


def _grid_row(cards: list[int], revealed: list[bool], row: int) -> str:
    """Render one row (3 slots) of a player's grid."""
    parts = []
    for col in range(3):
        slot = row * 3 + col
        if revealed[slot]:
            parts.append(f"{card_str(cards[slot]):>4}")
        else:
            parts.append("  ??")
    return "  ".join(parts)


def render_game_state(state: VectorizedGolfState, player_id: int) -> str:
    """Render the game state as compact text for the LLM."""
    device = state.player_cards.device

    own_cards = state.player_cards[0, player_id].tolist()
    own_rev = state.player_revealed[0, player_id].tolist()
    own_holding = state.player_holding[0, player_id].item()
    discard = state.discard_top[0].item()
    score = compute_score(
        state.player_cards[0:1, player_id, :],
        state.player_revealed[0:1, player_id, :],
        device,
    ).item()
    deck_remaining = max(0, 52 - state.deck_ptr[0].item())

    def slot(pos):
        return card_str(own_cards[pos]) if own_rev[pos] else "??"

    lines = [
        f"Your grid (score {score:.0f}):",
        f"  [{slot(0):>3} {slot(1):>3} {slot(2):>3}]",
        f"  [{slot(3):>3} {slot(4):>3} {slot(5):>3}]",
    ]

    if own_holding >= 0:
        lines.append(f"Holding: {card_str(own_holding)}")
    lines.append(f"Discard: {card_str(discard)}  Deck: {deck_remaining}")

    return "\n".join(lines)


def render_valid_actions(state: VectorizedGolfState, player_id: int, mask: torch.Tensor) -> str:
    """Enumerate valid actions with concise descriptions."""
    own_cards = state.player_cards[0, player_id].tolist()
    own_rev = state.player_revealed[0, player_id].tolist()
    holding = state.player_holding[0, player_id].item()
    discard = state.discard_top[0].item()
    stage = state.current_stage[0].item()

    valid_ids = [i for i in range(NUM_ACTIONS) if mask[i]]
    lines = [f"VALID ACTIONS (pick one of {valid_ids}):"]

    if stage == 0:
        if mask[0]:
            lines.append(f"  0 - Take discard card: {card_str(discard)}")
        if mask[1]:
            lines.append(f"  1 - Draw from deck")
    else:
        lines.append(f"  Holding: {card_str(holding)}")
        lines.append("")

        for pos in range(6):
            action_id = 2 + pos
            if mask[action_id]:
                row, col = pos // 3, pos % 3
                target = card_str(own_cards[pos]) if own_rev[pos] else "??"
                lines.append(f"  {action_id} - Place at Row {row}, Col {col} (swap with {target})")

        for pos in range(6):
            action_id = 9 + pos
            if mask[action_id]:
                row, col = pos // 3, pos % 3
                lines.append(f"  {action_id} - Discard and flip Row {row}, Col {col}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing Golf (card game). LOWEST score wins.

Scores: K=0, A=1, 2=-2, 3-9=face value, 10/J/Q=10.
If both cards in a column match rank, both score 0.
Your 6 cards are in a 2x3 grid. ?? = face-down (unknown).

Each turn: draw (take discard or draw deck), then place or discard.
Place = swap held card with any grid card. Discard = throw held card away and flip a face-down card.
Game ends when someone reveals all 6; others get one more turn. All cards revealed for final score.

Respond with ONLY your action number, e.g.:
Action: 3"""



# ---------------------------------------------------------------------------
# LLM Player
# ---------------------------------------------------------------------------

BACKENDS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key_env": None,
    },
}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models.
    If <think> is opened but never closed, strip everything from <think> onward."""
    import re
    # Strip closed thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip unclosed thinking block (truncated output)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_action(text: str) -> Optional[int]:
    """Extract action integer from LLM response text."""
    import re
    # Strip thinking blocks first
    text = _strip_thinking(text)
    # Try JSON: {"action": 3, ...}
    m = re.search(r'"action"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    # Try "Action: 3" or "action: 3"
    m = re.search(r'[Aa]ction\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    # Try bare number on last non-empty line
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


class LLMPlayer:
    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5",
        backend: str = "openrouter",
        temperature: float = 0.0,
    ):
        cfg = BACKENDS[backend]
        api_key = os.environ.get(cfg["api_key_env"] or "", "ollama")
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=120.0)
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.invalid_actions = 0
    def choose_action(self, state: VectorizedGolfState, player_id: int, verbose: bool = False) -> int:
        mask = get_valid_action_mask(state, player_id)[0]  # (16,) for N=1
        game_text = render_game_state(state, player_id)
        actions_text = render_valid_actions(state, player_id, mask)
        user_msg = f"{game_text}\n\n{actions_text}"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

        self.api_calls += 1
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        text = response.choices[0].message.content or ""
        action = _extract_action(text)
        valid = action is not None and 0 <= action < NUM_ACTIONS and mask[action]

        if not valid:
            self.invalid_actions += 1
            valid_indices = mask.nonzero(as_tuple=True)[0]
            action = valid_indices[torch.randint(len(valid_indices), (1,))].item()

        if verbose:
            stage = state.current_stage[0].item()
            tag = "ok" if valid else "INVALID"
            print(f"  [LLM] stage={stage} -> action {action} ({tag})")
            print(f"    state: {user_msg.splitlines()[0]}")
            display = _strip_thinking(text) or "(empty after stripping <think>)"
            for line in display.splitlines():
                print(f"    | {line}")

        return action

    def print_summary(self):
        print(f"\nAPI usage: {self.api_calls} calls, "
              f"{self.total_input_tokens:,} input tokens, "
              f"{self.total_output_tokens:,} output tokens")
        if self.invalid_actions > 0:
            print(f"Invalid actions (fell back to random): {self.invalid_actions}")


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def run_games(
    llm: LLMPlayer,
    seat_roles: List[str],
    num_games: int,
    holes: int,
    dqn_model=None,
    dqn_obs_fn=None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Tuple[Dict[int, float], Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    """Run multiple games, return (avg_per_hole, raw_totals, behavioral_metrics)."""

    totals = {i: np.zeros(num_games) for i in range(4)}

    # Behavioral accumulators
    s0_take = {i: 0 for i in range(4)}
    s0_turns = {i: 0 for i in range(4)}
    s1_place = {i: 0 for i in range(4)}
    s1_rev_replace = {i: 0 for i in range(4)}
    col_matches = {i: np.zeros(num_games) for i in range(4)}

    for game_idx in range(num_games):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Game {game_idx + 1}/{num_games}")
            print(f"{'='*60}")

        game_scores = {i: 0.0 for i in range(4)}

        for hole in range(1, holes + 1):
            state = reset_games(1, device)

            if verbose:
                print(f"\n--- Hole {hole}/{holes} ---", flush=True)
            else:
                print(f"  Hole {hole}/{holes}...", end=" ", flush=True)

            for round_num in range(30):
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

                    # -- Stage 0 --
                    state.current_stage.fill_(0)

                    if role == "llm":
                        a0 = llm.choose_action(state, pid, verbose=verbose)
                        actions_s0 = torch.tensor([a0], dtype=torch.long, device=device)
                    elif role == "heuristic":
                        actions_s0 = heuristic_stage0(state, pid)
                    elif role == "dqn":
                        obs = dqn_obs_fn(state, pid).to(device)
                        sg = torch.zeros(1, dtype=torch.long, device=device)
                        with torch.no_grad():
                            q = dqn_model(obs, sg)
                        s0_mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool, device=device)
                        s0_mask[:, 0] = True
                        s0_mask[:, 1] = state.deck_ptr < 52
                        from src.vectorized_golf import eps_greedy_batched
                        actions_s0 = eps_greedy_batched(q, 0.0, s0_mask)
                    else:  # random
                        from src.vectorized_golf import random_stage0
                        actions_s0 = random_stage0(state, pid)

                    # Track take rate
                    if active[0]:
                        s0_turns[pid] += 1
                        if actions_s0[0].item() == 0:
                            s0_take[pid] += 1

                    step_stage0(state, actions_s0, pid)
                    if state.done.all():
                        break

                    # -- Stage 1 --
                    state.current_stage.fill_(1)
                    was_revealed = state.player_revealed[:, pid, :].clone()

                    if role == "llm":
                        a1 = llm.choose_action(state, pid, verbose=verbose)
                        actions_s1 = torch.tensor([a1], dtype=torch.long, device=device)
                    elif role == "heuristic":
                        actions_s1 = heuristic_stage1(state, pid)
                    elif role == "dqn":
                        obs1 = dqn_obs_fn(state, pid).to(device)
                        sg1 = torch.ones(1, dtype=torch.long, device=device)
                        with torch.no_grad():
                            q1 = dqn_model(obs1, sg1)
                        mask1 = get_valid_action_mask(state, pid)
                        actions_s1 = eps_greedy_batched(q1, 0.0, mask1)
                    else:  # random
                        from src.vectorized_golf import random_stage1
                        actions_s1 = random_stage1(state, pid)

                    # Track place / rev_replace
                    if active[0]:
                        a = actions_s1[0].item()
                        if 2 <= a <= 7:
                            s1_place[pid] += 1
                            pos = a - 2
                            if was_revealed[0, pos]:
                                s1_rev_replace[pid] += 1

                    step_stage1(state, actions_s1, pid)

                    # End-game trigger
                    all_rev = state.player_revealed[:, pid, :].all(dim=1)
                    newly_last = active & all_rev & (~state.last_turn)
                    state.last_turn = state.last_turn | newly_last
                    state.end_game_player = torch.where(
                        newly_last,
                        torch.full_like(state.end_game_player, pid),
                        state.end_game_player,
                    )

            # Score this hole
            hole_scores = {}
            for sid in range(4):
                hole_score = compute_final_score(
                    state.player_cards[:, sid, :], device
                ).item()
                game_scores[sid] += hole_score
                hole_scores[sid] = hole_score
                col_matches[sid][game_idx] += count_column_matches(state, sid).item()

            if not verbose:
                llm_sid = seat_roles.index("llm") if "llm" in seat_roles else 0
                print(f"score {hole_scores[llm_sid]:.0f}", flush=True)

        # Accumulate game totals
        for sid in range(4):
            totals[sid][game_idx] = game_scores[sid]

        # Print per-game summary
        parts = []
        for sid in range(4):
            label = seat_roles[sid].upper()
            avg = game_scores[sid] / holes
            parts.append(f"{label}={avg:.1f}")
        print(f"  Game {game_idx + 1:>3}/{num_games}: {', '.join(parts)}", flush=True)

    # Compute behavioral metrics
    behavior = {}
    for sid in range(4):
        take_rate = s0_take[sid] / max(s0_turns[sid], 1)
        rev_rate = s1_rev_replace[sid] / max(s1_place[sid], 1)
        avg_col = col_matches[sid].mean() / holes
        behavior[sid] = {
            "take_rate": round(take_rate, 3),
            "rev_replace": round(rev_rate, 3),
            "col_matches": round(avg_col, 3),
        }

    avgs = {sid: totals[sid].mean() / holes for sid in range(4)}
    return avgs, totals, behavior


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Golf player via OpenRouter")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-5",
                        help="Model ID (OpenRouter or Ollama model name)")
    parser.add_argument("--backend", default="openrouter", choices=list(BACKENDS),
                        help="API backend: openrouter or ollama")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--holes", type=int, default=9)
    parser.add_argument("--seats", default="llm,random,heuristic,random",
                        help="Comma-separated seat roles: llm, random, heuristic, dqn")
    parser.add_argument("--dqn-checkpoint", type=str, default=None,
                        help="Path to DQN checkpoint .pt file")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="data/llm_benchmarks",
                        help="Directory for result JSON files")
    args = parser.parse_args()

    seat_roles = [s.strip() for s in args.seats.split(",")]
    assert len(seat_roles) == 4, "Need exactly 4 seat roles"

    from src.dqn_offline import resolve_device
    device = resolve_device(args.device)

    # Load DQN if needed
    dqn_model = None
    dqn_obs_fn = None
    if "dqn" in seat_roles:
        assert args.dqn_checkpoint, "--dqn-checkpoint required when using dqn seats"
        from src.tournament import make_model, get_obs_fn
        ckpt = torch.load(args.dqn_checkpoint, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        variant = cfg.get("model_variant", "v1")
        dqn_model = make_model(variant, cfg.get("embedding_dim", 128), cfg["hidden_dim"], device)
        dqn_model.load_state_dict(ckpt["model_state_dict"])
        dqn_model.eval()
        dqn_obs_fn = get_obs_fn(variant)
        print(f"Loaded DQN: {ckpt.get('agent_record', {}).get('agent_id', '?')}")

    llm = LLMPlayer(
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
    )

    print(f"Model: {args.model} ({args.backend})")
    print(f"Seats: {seat_roles}")
    print(f"Games: {args.games} x {args.holes} holes")
    print(flush=True)

    t0 = time.time()
    avgs, totals, behavior = run_games(
        llm, seat_roles, args.games, args.holes,
        dqn_model=dqn_model, dqn_obs_fn=dqn_obs_fn,
        device=device, verbose=args.verbose,
    )
    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*60}")
    print(f"RESULTS ({args.games} games x {args.holes} holes, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  {'seat':<12}  {'avg/hole':>9}  {'std':>6}  {'median':>8}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*6}  {'-'*8}")
    for sid in range(4):
        per_hole = totals[sid] / args.holes
        label = seat_roles[sid].upper()
        if seat_roles[sid] == "llm":
            label = f"LLM"
        print(f"  {label:<12}  {avgs[sid]:>9.2f}  {per_hole.std():>6.2f}  {np.median(per_hole):>8.2f}")

    # Behavioral metrics
    print(f"\n  {'seat':<12}  {'take_rate':>10}  {'rev_replace':>12}  {'col_matches':>12}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*12}")
    for sid in range(4):
        b = behavior[sid]
        label = seat_roles[sid].upper()
        if seat_roles[sid] == "llm":
            label = "LLM"
        print(f"  {label:<12}  {b['take_rate']:>10.3f}  {b['rev_replace']:>12.3f}  {b['col_matches']:>12.3f}")

    llm.print_summary()

    # Save results to JSON
    from pathlib import Path
    from datetime import datetime
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_slug = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": args.model,
        "backend": args.backend,
        "games": args.games,
        "holes": args.holes,
        "seats": seat_roles,
        "temperature": args.temperature,
        "elapsed_seconds": round(elapsed, 1),
        "api_calls": llm.api_calls,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "invalid_actions": llm.invalid_actions,
        "scores": {seat_roles[sid]: {
            "avg_per_hole": round(avgs[sid], 2),
            "std": round((totals[sid] / args.holes).std(), 2),
            "median": round(float(np.median(totals[sid] / args.holes)), 2),
        } for sid in range(4)},
        "behavior": {seat_roles[sid]: behavior[sid] for sid in range(4)},
        "per_game_totals": {seat_roles[sid]: totals[sid].tolist() for sid in range(4)},
    }

    out_path = out_dir / f"{model_slug}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""Capture a real-game L vs L2 end-of-hole timing decision.

The artifact deliberately records only the acting player's observable state;
the latent value of the final hidden card is not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.seat_cycling import SeatHandler
from src.vectorized_golf import compute_final_score, reset_games, step_stage0, step_stage1


ACTION_NAMES = {
    **{2 + p: f"place slot {p}" for p in range(6)},
    **{9 + p: f"discard and flip slot {p}" for p in range(6)},
}


def observable(state, row: int, player_id: int) -> dict:
    cards = state.player_cards[row, player_id].tolist()
    revealed = state.player_revealed[row, player_id].tolist()
    return {
        "cards": [card if visible else "hidden" for card, visible in zip(cards, revealed)],
        "revealed": revealed,
        "holding": int(state.player_holding[row, player_id]),
        "discard_top": int(state.discard_top[row]),
        "last_turn": bool(state.last_turn[row]),
        "deck_remaining": int((state.deck_size[row] - state.deck_ptr[row]).clamp(min=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    N = args.games
    handlers = [
        SeatHandler("L2", 0, N, device),
        SeatHandler("R", 1, N, device),
        SeatHandler("R", 2, N, device),
        SeatHandler("R", 3, N, device),
    ]
    shadow_l = SeatHandler("L", 0, N, device)
    state = reset_games(N, device, n_players=4)
    for handler in handlers:
        handler.reset_for_hole(state)
    shadow_l.reset_for_hole(state)

    found = None
    returned = None
    capture_row = None

    for _round in range(60):
        if state.done.all():
            break
        for pid, handler in enumerate(handlers):
            active = ~state.done
            back_to_trigger = state.last_turn & (state.end_game_player == pid)
            state.done = state.done | (back_to_trigger & active)
            active = ~state.done
            if not active.any():
                break

            if pid == 0 and capture_row is not None and returned is None:
                returned = observable(state, capture_row, 0)

            state.current_stage.fill_(0)
            actual_a0 = handler.stage0(state)
            if pid == 0:
                shadow_l.stage0(state)
            step_stage0(state, actual_a0, pid)
            for h in handlers:
                h.observe(state)
            shadow_l.observe(state)

            if state.done.all():
                break

            state.current_stage.fill_(1)
            l_action = None
            if pid == 0:
                l_action = shadow_l.stage1(state)
            actual_a1 = handler.stage1(state)

            if pid == 0 and found is None:
                for row in range(N):
                    if int(l_action[row]) != int(actual_a1[row]):
                        found = {
                            "row": row,
                            "round": _round,
                            "before_action": observable(state, row, 0),
                            "l_action": int(l_action[row]),
                            "l_action_name": ACTION_NAMES.get(int(l_action[row]), str(int(l_action[row]))),
                            "l2_action": int(actual_a1[row]),
                            "l2_action_name": ACTION_NAMES.get(int(actual_a1[row]), str(int(actual_a1[row]))),
                        }
                        capture_row = row
                        break

            step_stage1(state, actual_a1, pid)
            for h in handlers:
                h.observe(state)
            shadow_l.observe(state)

            all_rev = state.player_revealed[:, pid, :].all(dim=1)
            newly_last = active & all_rev & (~state.last_turn)
            state.last_turn = state.last_turn | newly_last
            state.end_game_player = torch.where(
                newly_last,
                torch.full_like(state.end_game_player, pid),
                state.end_game_player,
            )

    if found is None or returned is None or capture_row is None:
        raise RuntimeError("No L/L2 timing disagreement found; increase --games")

    found["next_turn_state"] = returned
    found["final_score_after_l2_path"] = float(
        compute_final_score(state.player_cards[capture_row, 0].unsqueeze(0), device)[0]
    )
    artifact = {
        "seed": args.seed,
        "games": args.games,
        "roster": ["L2", "R", "R", "R"],
        "demonstration": found,
        "interpretation": (
            "L would reveal the final hidden card and finish this player's hole. "
            "L2 places into a revealed slot instead; the next_turn_state shows "
            "that one hidden card remains and the player receives another turn."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

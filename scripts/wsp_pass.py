"""One forward pass for the weight-space programming experiment (WSP).

The programmer (an LLM) edits the DQN champion's weights by hand. The only way
it may see the network play is this script: each call scores one candidate
checkpoint in seat-cycled C,L,I,R games on the practice seed and spends one
pass of the run's budget. It also writes a trace of the first few games so the
programmer can look at how the candidate loses.

    python -m scripts.wsp_pass data/wsp/<run>/cand_03.pt --run-dir data/wsp/<run>

Held-out scoring (operator only, no budget): seatcycle-v1 on seed 0 through
service.runner.execute, so it matches the leaderboard rows.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path

import torch

import scripts.seat_cycling as sc
from service.runner import execute as ex

# Seed 0 is the held-out seed (seatcycle-v1/quick); practice games must not share its decks.
ex.PROTOCOLS["wsp-practice"] = {"games_per_perm": 100, "holes": 9, "seed": 1}


def check_candidate(cand: Path, base: Path) -> None:
    """Same architecture as the base, same tensor shapes, all finite."""
    c = torch.load(cand, map_location="cpu", weights_only=False)
    b = torch.load(base, map_location="cpu", weights_only=False)
    if c.get("config") != b.get("config"):
        raise SystemExit(f"config differs from base: {c.get('config')} vs {b.get('config')}")
    cs, bs = c["model_state_dict"], b["model_state_dict"]
    if cs.keys() != bs.keys():
        raise SystemExit(f"state_dict keys differ from base: {sorted(cs.keys() ^ bs.keys())}")
    for k in bs:
        if cs[k].shape != bs[k].shape:
            raise SystemExit(f"{k}: shape {tuple(cs[k].shape)} != base {tuple(bs[k].shape)}")
        if cs[k].is_floating_point() and not torch.isfinite(cs[k]).all():
            raise SystemExit(f"{k}: non-finite values")


class Tracer:
    """Records the candidate's decisions and hole scores for the first K games of the first seating."""

    def __init__(self, k: int):
        self.k, self.seating, self.rows, self.hole = k, -1, [], 0
        self._obs = self._stage = None
        self._scores: list = []
        self.seat_c = None

    def on(self) -> bool:
        return self.seating == 0 and self.k > 0

    def install(self) -> None:
        run_seating, eps, final = sc.run_seating, sc.eps_greedy_batched, sc.compute_final_score
        tr = self

        def run_seating_w(seating, *a, **kw):
            tr.seating += 1
            tr.hole = 0
            if tr.seating == 0:
                tr.seat_c = list(seating).index("C")
                tr.rows.append({"type": "seating", "seating": list(seating), "candidate_seat": tr.seat_c})
            return run_seating(seating, *a, **kw)

        def eps_w(q, epsilon, mask):
            a = eps(q, epsilon, mask)
            if tr.on() and tr._obs is not None:
                for g in range(min(tr.k, q.shape[0])):
                    tr.rows.append({
                        "type": "decision", "game": g, "hole": tr.hole, "stage": tr._stage,
                        "obs": tr._obs[g].tolist(),
                        "q": [round(x, 3) for x in q[g].tolist()],
                        "valid": mask[g].nonzero().flatten().tolist(), "action": int(a[g]),
                    })
            tr._obs = None
            return a

        def final_w(cards, device):
            s = final(cards, device)
            if tr.on():
                tr._scores.append(s[: tr.k].tolist())
                if len(tr._scores) == 4:  # one call per seat at hole end
                    for g in range(min(tr.k, len(s))):
                        tr.rows.append({"type": "hole_end", "game": g, "hole": tr.hole,
                                        "scores_by_seat": [tr._scores[sid][g] for sid in range(4)]})
                    tr._scores, tr.hole = [], tr.hole + 1
            return s

        sc.run_seating, sc.eps_greedy_batched, sc.compute_final_score = run_seating_w, eps_w, final_w

        # Wrap the registered model so we see the obs it was given.
        model, obs_fn, dev = sc.SeatHandler.dqn_registry["C"]

        class Rec(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, obs, stage):
                if tr.on():
                    tr._obs, tr._stage = obs[: tr.k].cpu(), int(stage[0])
                return self.m(obs, stage)

        sc.SeatHandler.dqn_registry["C"] = (Rec(model), obs_fn, dev)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--trace-games", type=int, default=8)
    args = p.parse_args(argv)

    budget = json.loads((args.run_dir / "budget.json").read_text())
    ledger = args.run_dir / "passes.jsonl"
    used = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    if used >= budget["max_passes"]:
        raise SystemExit(f"budget spent: {used}/{budget['max_passes']} passes used")
    ex.PROTOCOLS["wsp-practice"]["seed"] = budget.get("practice_seed", 1)
    check_candidate(args.candidate, args.run_dir / "base.pt")

    torch.set_num_threads(4)
    ex.load_dqn(args.candidate)
    ex.CandidateHandler.candidate = types.SimpleNamespace(kind="dqn")
    tracer = Tracer(args.trace_games)
    tracer.install()
    t0 = time.time()
    res = ex.evaluate("wsp-practice", torch.device("cpu"))

    n = used + 1
    trace_path = args.run_dir / f"trace_{n:03d}.jsonl"
    trace_path.write_text("".join(json.dumps(r) + "\n" for r in tracer.rows))
    entry = {
        "pass": n, "candidate": str(args.candidate), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avg_score_per_hole": res["metric"]["value"], "win_rate": res["win_rate"],
        "per_label": res["per_label"], "per_seating": res["per_seating"],
        "trace": str(trace_path), "seconds": round(time.time() - t0, 1),
    }
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    # Paired vs pass 1 (the base): same seed and seatings, so seat effects cancel per seating.
    cand = {tuple(r["seating"]): r["avg"][r["seating"].index("C")] for r in res["per_seating"]}
    first = json.loads(ledger.read_text().splitlines()[0])
    base = {tuple(r["seating"]): r["avg"][r["seating"].index("C")] for r in first["per_seating"]}
    d = [cand[k] - base[k] for k in cand]
    md = sum(d) / len(d)
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else 0.0
    print(f"pass {n}/{budget['max_passes']}  avg_score_per_hole={entry['avg_score_per_hole']:.4f} "
          f"(lower is better)  win_rate={entry['win_rate']:.4f}")
    print(f"vs pass 1 ({first['candidate']}): {md:+.4f} per hole, paired SE {se:.4f} over {len(d)} seatings")
    print("per_label:", json.dumps(res["per_label"]))
    print("trace:", trace_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

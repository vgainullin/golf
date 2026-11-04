"""Forward passes for the weight-space programming experiment (WSP).

The programmer (an LLM) edits the DQN champion's weights by hand. The only way
it may see the network play is this script. Each call scores one candidate
checkpoint as C in seat-cycled C,L,I,R games and spends one unit of a budget.
Every tier deals identical cards per (seating, hole) to every candidate (common
random numbers), so paired differences against the base measure decisions, not
luck.

Tiers (budgets and sizes in <run>/budget.json):
  practice  (default)  cheap screening pass on the practice seed, with a trace of
                       games from every seating so the programmer can see how the
                       candidate loses.
  confirm   --confirm  a larger pass on a separate seed and a small budget. The
                       run's final.pt must be byte-identical to a confirmed
                       candidate; this guards against shipping the best of many
                       near-ties picked on practice noise.
  baseline  --baseline operator only: scores base.pt on both tiers once, free,
                       before the programmer starts. Paired differences are
                       against these.

    python -m scripts.wsp_pass data/wsp/<run>/cand_03.pt --run-dir data/wsp/<run> [--confirm]

Held-out scoring (operator only): seatcycle-v1 on seed 0 through
service.runner.execute, so it matches the leaderboard rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import types
from pathlib import Path

import torch

import scripts.seat_cycling as sc
from service.runner import execute as ex

# Held-out is seatcycle-v1 (seed 0, no common deals); practice and confirm deal from their own seeds.
DEFAULTS = {"practice_seed": 1, "confirm_seed": 2, "practice_games_per_perm": 300,
            "confirm_games_per_perm": 1000, "max_confirms": 3, "trace_per_seating": 2, "holes": 9}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Records the candidate's decisions and hole scores for the first k games of every seating.

    Decisions in games that are already over are dropped (the batch keeps querying the net for
    them). Each decision carries the candidate's history-aware belief: the count of each rank it
    has not seen this hole, from the same tracker the lookahead uses, run from C's seat.
    """

    def __init__(self, k: int):
        self.k, self.rows = k, []
        self.seating, self.hole = -1, 0
        self._obs = self._stage = self._state = self._belief = None
        self._scores: list = []

    def install(self) -> None:
        tr = self
        run_seating, eps, final = sc.run_seating, sc.eps_greedy_batched, sc.compute_final_score
        H = ex.CandidateHandler
        h_reset, h_observe, h_cand = H.reset_for_hole, H.observe, H._cand

        def run_seating_w(seating, *a, **kw):
            tr.seating += 1
            tr.hole = 0
            tr.rows.append({"type": "seating", "seating_index": tr.seating, "seating": list(seating),
                            "candidate_seat": list(seating).index("C")})
            return run_seating(seating, *a, **kw)

        def reset_w(self, state):
            if self.label == "C":
                tr._belief = sc.BayesBeliefTracker(self.N, self.device)
                tr._belief.observe(state, my_player_id=self.seat)
            return h_reset(self, state)

        def observe_w(self, state):
            if self.label == "C":
                tr._belief.observe(state, my_player_id=self.seat)
            return h_observe(self, state)

        def cand_w(self, state, stage):
            tr._state = state
            return h_cand(self, state, stage)

        def eps_w(q, epsilon, mask):
            a = eps(q, epsilon, mask)
            if tr._obs is not None:
                k = min(tr.k, q.shape[0])
                unseen = tr._belief.unobserved[:k].reshape(k, 4, 13).sum(1)  # card index % 13 = rank
                exp_unknown = tr._belief.expected_unknown_score()[:k]
                for g in range(k):
                    if tr._state.done[g]:
                        continue
                    tr.rows.append({
                        "type": "decision", "seating_index": tr.seating, "game": g, "hole": tr.hole,
                        "stage": tr._stage, "obs": tr._obs[g].tolist(),
                        "q": [round(x, 3) for x in q[g].tolist()],
                        "valid": mask[g].nonzero().flatten().tolist(), "action": int(a[g]),
                        "unseen_by_rank": unseen[g].tolist(),
                        "expected_unknown_score": round(float(exp_unknown[g]), 3),
                    })
            tr._obs = None
            return a

        def final_w(cards, device):
            s = final(cards, device)
            tr._scores.append(s[: tr.k].tolist())
            if len(tr._scores) == 4:  # one call per seat at hole end
                for g in range(len(tr._scores[0])):
                    tr.rows.append({"type": "hole_end", "seating_index": tr.seating, "game": g,
                                    "hole": tr.hole, "scores_by_seat": [tr._scores[sid][g] for sid in range(4)]})
                tr._scores, tr.hole = [], tr.hole + 1
            return s

        sc.run_seating, sc.eps_greedy_batched, sc.compute_final_score = run_seating_w, eps_w, final_w
        H.reset_for_hole, H.observe, H._cand = reset_w, observe_w, cand_w

        # Wrap the registered model so we see the obs it was given.
        model, obs_fn, dev = sc.SeatHandler.dqn_registry["C"]

        class Rec(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, obs, stage):
                tr._obs, tr._stage = obs[: tr.k].cpu(), int(stage[0])
                return self.m(obs, stage)

        sc.SeatHandler.dqn_registry["C"] = (Rec(model), obs_fn, dev)


def score(candidate: Path, seed: int, games_per_perm: int, trace_k: int, holes: int = 9) -> tuple[dict, list]:
    """Seat-cycled C,L,I,R with common deals; returns (runner result, trace rows)."""
    torch.set_num_threads(4)
    ex.PROTOCOLS["wsp"] = {"games_per_perm": games_per_perm, "holes": holes, "seed": seed, "crn": True}
    ex.load_dqn(candidate)
    ex.CandidateHandler.candidate = types.SimpleNamespace(kind="dqn")
    tracer = Tracer(trace_k)
    if trace_k:
        tracer.install()
    return ex.evaluate("wsp", torch.device("cpu")), tracer.rows


def c_by_seating(res: dict) -> dict:
    return {tuple(r["seating"]): r["avg"][r["seating"].index("C")] for r in res["per_seating"]}


def paired(res: dict, base: dict) -> tuple[float, float]:
    """Mean and SE of the per-seating difference candidate − base."""
    c, b = c_by_seating(res), c_by_seating(base)
    d = [c[k] - b[k] for k in c]
    m = sum(d) / len(d)
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path, nargs="?")
    p.add_argument("--run-dir", type=Path, required=True)
    tier = p.add_mutually_exclusive_group()
    tier.add_argument("--confirm", action="store_true", help="spend a confirm pass")
    tier.add_argument("--baseline", action="store_true", help="operator: score base.pt on both tiers")
    args = p.parse_args(argv)
    run = args.run_dir
    cfg = {**DEFAULTS, **json.loads((run / "budget.json").read_text())}
    tiers = {"practice": (cfg["practice_seed"], cfg["practice_games_per_perm"]),
             "confirm": (cfg["confirm_seed"], cfg["confirm_games_per_perm"])}

    if args.baseline:
        for name, (seed, gpp) in tiers.items():
            out = run / f"base_{name}.json"
            if out.exists():
                raise SystemExit(f"{out} already exists")
            t0 = time.time()
            res, _ = score(run / "base.pt", seed, gpp, 0, cfg["holes"])
            out.write_text(json.dumps({"sha256": sha256(run / "base.pt"), "seconds": round(time.time() - t0, 1),
                                       **res}))
            print(f"base {name}: {res['metric']['value']:.4f} per hole, win {res['win_rate']:.4f} "
                  f"({out.name}, {time.time() - t0:.0f}s)")
        return 0

    if args.candidate is None:
        raise SystemExit("candidate checkpoint required")
    name = "confirm" if args.confirm else "practice"
    base_res_path = run / f"base_{name}.json"
    if not base_res_path.exists():
        raise SystemExit(f"{base_res_path} missing: the operator runs --baseline before the session")
    ledger = run / ("confirms.jsonl" if args.confirm else "passes.jsonl")
    limit = cfg["max_confirms"] if args.confirm else cfg["max_passes"]
    used = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    check_candidate(args.candidate, run / "base.pt")
    if used >= limit:
        raise SystemExit(f"{name} budget spent: {used}/{limit} used")

    seed, gpp = tiers[name]
    t0 = time.time()
    res, rows = score(args.candidate, seed, gpp, 0 if args.confirm else cfg["trace_per_seating"], cfg["holes"])
    n = used + 1
    md, se = paired(res, json.loads(base_res_path.read_text()))
    entry = {
        "pass": n, "tier": name, "candidate": str(args.candidate), "sha256": sha256(args.candidate),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
        "avg_score_per_hole": res["metric"]["value"], "win_rate": res["win_rate"],
        "vs_base": round(md, 4), "vs_base_se": round(se, 4),
        "per_label": res["per_label"], "per_seating": res["per_seating"],
    }
    if rows:
        trace_path = run / f"trace_{n:03d}.jsonl"
        trace_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        entry["trace"] = str(trace_path)
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"{name} {n}/{limit}  avg_score_per_hole={entry['avg_score_per_hole']:.4f} "
          f"(lower is better)  win_rate={entry['win_rate']:.4f}")
    print(f"vs base.pt: {md:+.4f} per hole, paired SE {se:.4f} over {len(res['per_seating'])} seatings")
    print("per_label:", json.dumps(res["per_label"]))
    if rows:
        print("trace:", entry["trace"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

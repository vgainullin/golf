"""Execute one research run for the golf project and write a result JSON.

This is the compute half of the research-run service. The Worker API decides
*what* to run (a run spec, written by the research agent or a donor) and this
script does the work: optionally train a candidate, then score it with the
fixed seat-cycling benchmark.

    python -m service.runner.execute --spec spec.json --out result.json

Spec kinds:

  builtin     Evaluate one of golf's existing players by label (I, H, L, R).
              Used to seed baselines for a protocol.
  agent_code  Python source defining stage0(state, seat) and stage1(state, seat),
              each returning an (N,) long tensor of actions. Optional hooks:
              reset(state, seat) at the start of each hole and observe(state, seat)
              after every step by any player (for belief tracking). Evaluated as label C.
  checkpoint  A DQN checkpoint (.pt) at a local path or http(s) URL.
  train       Run src.tournament with whitelisted flags, then evaluate the
              resulting champion.pt.

Benchmark: the candidate takes seat label C in the roster C,L,I,R and plays every
distinct seating (24) for `games_per_perm` games of `holes` holes. The metric is
the candidate's average score per hole (lower is better), with win rate alongside.
This is the same protocol as the DQN champion row in the README (it played as D
in L,D,I,R), so the numbers line up with data/seat_cycling_exp14_vs_lookahead.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
import types
import urllib.request
from pathlib import Path

import torch

import scripts.seat_cycling as sc

ROSTER_OPPONENTS = ["L", "I", "R"]
BUILTIN_LABELS = {"I", "H", "L", "R", "B2", "B3"}

# Protocols are fixed so leaderboard rows are comparable. A run names one.
PROTOCOLS = {
    "seatcycle-v1": {"games_per_perm": 1000, "holes": 9, "seed": 0},
    "seatcycle-quick": {"games_per_perm": 100, "holes": 9, "seed": 0},
}

# Tournament flags a spec may set. Anything else is rejected so a spec can't
# point output at odd places or turn on uploads.
TRAIN_FLAGS = {
    "population_size": int, "generations": int, "episodes_per_gen": int,
    "updates_per_episode": int, "batch_size": int, "buffer_capacity": int,
    "target_update_interval": int, "epsilon_start": float, "epsilon_end": float,
    "cycle_length": int, "gamma": float, "lr_range": list, "embedding_dim": int,
    "hidden_dim_choices": list, "model_variant": str, "n_step": int,
    "reward_shaping": str, "win_bonus": float, "loss_penalty": float, "seed": int,
}
# Hard caps for CPU runs (GitHub-hosted runners, local laptops).
CPU_TRAIN_CAPS = {"population_size": 8, "generations": 30, "episodes_per_gen": 1000}


class CandidateHandler(sc.SeatHandler):
    """SeatHandler that also understands label C, the candidate under test."""

    LABELS = sc.SeatHandler.LABELS + ("C",)
    candidate: "types.SimpleNamespace | None" = None  # .kind, .stage0, .stage1

    def __init__(self, label, seat_idx, N, device):
        if label == "C":
            self.label, self.seat, self.N, self.device = label, seat_idx, N, device
            self.tracker = None
            self._inner = None
            c = CandidateHandler.candidate
            if c.kind == "builtin":
                # Delegate to a stock handler for that label (keeps its tracker).
                self._inner = sc.SeatHandler(c.label, seat_idx, N, device)
                self.tracker = self._inner.tracker
        else:
            super().__init__(label, seat_idx, N, device)

    def reset_for_hole(self, state):
        if self.label == "C" and self._inner is not None:
            return self._inner.reset_for_hole(state)
        if self.label == "C" and hasattr(CandidateHandler.candidate, "reset"):
            CandidateHandler.candidate.reset(state, self.seat)
        return super().reset_for_hole(state)

    def observe(self, state):
        if self.label == "C" and self._inner is not None:
            return self._inner.observe(state)
        if self.label == "C" and hasattr(CandidateHandler.candidate, "observe"):
            CandidateHandler.candidate.observe(state, self.seat)
        return super().observe(state)

    def _cand(self, state, stage):
        c = CandidateHandler.candidate
        if c.kind == "builtin":
            return self._inner.stage0(state) if stage == 0 else self._inner.stage1(state)
        if c.kind == "dqn":
            return self._dqn_action(state, stage)
        fn = c.stage0 if stage == 0 else c.stage1
        return torch.as_tensor(fn(state, self.seat), dtype=torch.long).cpu()

    def stage0(self, state):
        return self._cand(state, 0) if self.label == "C" else super().stage0(state)

    def stage1(self, state):
        return self._cand(state, 1) if self.label == "C" else super().stage1(state)


def load_dqn(path: Path) -> None:
    from src.tournament import get_obs_fn, make_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or ckpt.get("hyperparams", {})
    variant = cfg.get("model_variant", "v1")
    model = make_model(variant, cfg.get("embedding_dim", 128), cfg.get("hidden_dim", 256), torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # _dqn_action looks the model up by the handler's label.
    sc.SeatHandler.dqn_registry["C"] = (model, get_obs_fn(variant), torch.device("cpu"))


def fetch(url_or_path: str, dest: Path) -> Path:
    if url_or_path.startswith(("http://", "https://")):
        urllib.request.urlretrieve(url_or_path, dest)
        return dest
    return Path(url_or_path)


def train(spec: dict, workdir: Path, device: str) -> tuple[Path, dict]:
    args = spec.get("train", {}).get("args", {})
    unknown = set(args) - set(TRAIN_FLAGS)
    if unknown:
        raise ValueError(f"unsupported train flags: {sorted(unknown)}")
    if device == "cpu":
        for k, cap in CPU_TRAIN_CAPS.items():
            if k in args and int(args[k]) > cap:
                raise ValueError(f"{k}={args[k]} exceeds the CPU cap of {cap}; use GPU compute")
    out_dir = workdir / "train"
    cmd = [sys.executable, "-u", "-m", "src.tournament", "--output-dir", str(out_dir), "--device", device]
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, list):
            cmd += [flag, *[str(x) for x in v]]
        else:
            cmd += [flag, str(v)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    info = {"cmd": cmd, "seconds": round(time.time() - t0, 1), "returncode": proc.returncode,
            "log_tail": (proc.stdout + proc.stderr)[-4000:]}
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (exit {proc.returncode})\n{info['log_tail']}")
    champ = out_dir / "champion.pt"
    if not champ.exists():
        raise RuntimeError("training finished but wrote no champion.pt")
    return champ, info


def build_candidate(spec: dict, workdir: Path, device: str) -> tuple[types.SimpleNamespace, dict]:
    kind = spec["kind"]
    extra: dict = {}
    if kind == "builtin":
        label = spec["builtin"]["label"]
        if label not in BUILTIN_LABELS:
            raise ValueError(f"builtin label must be one of {sorted(BUILTIN_LABELS)}")
        return types.SimpleNamespace(kind="builtin", label=label), extra
    if kind == "agent_code":
        # Runs donor/agent supplied code. The runner must not hold secrets in
        # this process's environment (the GitHub workflow keeps the callback
        # token in a later step).
        mod = types.ModuleType("candidate_agent")
        exec(compile(spec["agent_code"], "candidate_agent.py", "exec"), mod.__dict__)
        for fn in ("stage0", "stage1"):
            if not callable(getattr(mod, fn, None)):
                raise ValueError(f"agent_code must define {fn}(state, seat)")
        hooks = {h: getattr(mod, h) for h in ("reset", "observe") if callable(getattr(mod, h, None))}
        return types.SimpleNamespace(kind="code", stage0=mod.stage0, stage1=mod.stage1, **hooks), extra
    if kind == "checkpoint":
        path = fetch(spec["checkpoint"]["url"], workdir / "candidate.pt")
        load_dqn(path)
        return types.SimpleNamespace(kind="dqn"), extra
    if kind == "train":
        champ, info = train(spec, workdir, device)
        load_dqn(champ)
        extra["train"] = info
        extra["champion_path"] = str(champ)
        return types.SimpleNamespace(kind="dqn"), extra
    raise ValueError(f"unknown spec kind {kind!r}")


def evaluate(protocol_id: str, device: torch.device, games_override: int | None = None) -> dict:
    proto = dict(PROTOCOLS[protocol_id])
    if games_override:
        proto["games_per_perm"] = games_override
    torch.manual_seed(proto["seed"])
    sc.SeatHandler = CandidateHandler  # run_seating builds handlers by this name
    roster = ["C", *ROSTER_OPPONENTS]
    t0 = time.time()
    means, wins, per_perm = sc.run_matchup(roster, proto["games_per_perm"], proto["holes"], device)
    return {
        "protocol": {"id": protocol_id, "roster": roster, **proto},
        "metric": {"name": "avg_score_per_hole", "value": round(means["C"], 4), "lower_is_better": True},
        "win_rate": round(wins["C"], 4),
        "per_label": {k: {"avg_score_per_hole": round(means[k], 4), "win_rate": round(wins[k], 4)} for k in means},
        "per_seating": [
            {"seating": list(s), "avg": [round(x, 3) for x in a], "win": [round(x, 4) for x in w]}
            for s, a, w in per_perm
        ],
        "eval_seconds": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="path to run spec JSON")
    p.add_argument("--out", required=True, help="where to write result JSON")
    p.add_argument("--device", default="cpu")
    p.add_argument("--games-per-perm", type=int, default=None,
                   help="override the protocol's game count (debug only; the result is marked)")
    args = p.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text())
    protocol_id = spec.get("protocol", "seatcycle-quick")
    if protocol_id not in PROTOCOLS:
        raise SystemExit(f"unknown protocol {protocol_id!r}")
    result: dict
    with tempfile.TemporaryDirectory() as tmp:
        try:
            candidate, extra = build_candidate(spec, Path(tmp), args.device)
            CandidateHandler.candidate = candidate
            result = {"status": "ok", **evaluate(protocol_id, torch.device(args.device), args.games_per_perm), **extra}
            if args.games_per_perm:
                result["protocol"]["overridden"] = True
        except Exception as e:  # report the failure back instead of crashing silently
            result = {"status": "error", "error": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()[-4000:]}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("status", "metric", "win_rate", "error")}))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

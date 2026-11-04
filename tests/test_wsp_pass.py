"""WSP forward-pass harness: baselines, budgets, common deals, traces. Uses a random v3 net and
tiny game counts; each scenario runs in a subprocess because the tracer patches module state."""
import json
import subprocess
import sys

import pytest
import torch

from src.tournament import make_model

CFG = {"model_variant": "v3", "embedding_dim": 64, "hidden_dim": 256}


def _save(model, path):
    torch.save({"model_state_dict": model.state_dict(), "config": dict(CFG)}, path)


def _pass(run, *args):
    return subprocess.run([sys.executable, "-m", "scripts.wsp_pass", *args, "--run-dir", str(run)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    run = tmp_path_factory.mktemp("wsp")
    torch.manual_seed(0)
    m = make_model("v3", 64, 256, torch.device("cpu"))
    _save(m, run / "base.pt")
    with torch.no_grad():
        m.mlp[4].bias[1] += 1.0  # a real edit: prefer drawing
    _save(m, run / "cand.pt")
    (run / "budget.json").write_text(json.dumps({
        "max_passes": 2, "max_confirms": 1, "practice_games_per_perm": 2,
        "confirm_games_per_perm": 3, "trace_per_seating": 1, "holes": 2}))
    return run


def _ledger(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_flow(run):
    # The programmer cannot pass before the operator scores the base.
    r = _pass(run, str(run / "base.pt"))
    assert r.returncode != 0 and "--baseline" in r.stderr

    assert _pass(run, "--baseline").returncode == 0
    assert (run / "base_practice.json").exists() and (run / "base_confirm.json").exists()
    assert _pass(run, "--baseline").returncode != 0  # never overwritten

    # Common deals: re-scoring the base reproduces the baseline exactly.
    assert _pass(run, str(run / "base.pt")).returncode == 0
    e = _ledger(run / "passes.jsonl")[0]
    assert e["vs_base"] == 0 and e["vs_base_se"] == 0

    assert _pass(run, str(run / "cand.pt")).returncode == 0
    r = _pass(run, str(run / "cand.pt"))
    assert r.returncode != 0 and "budget spent" in r.stderr

    # Confirm tier has its own ledger and budget, and records the file hash.
    assert _pass(run, str(run / "cand.pt"), "--confirm").returncode == 0
    c = _ledger(run / "confirms.jsonl")
    assert len(c) == 1 and c[0]["tier"] == "confirm" and len(c[0]["sha256"]) == 64
    assert _pass(run, str(run / "cand.pt"), "--confirm").returncode != 0


def test_trace(run):
    rows = [json.loads(l) for l in (run / "trace_001.jsonl").read_text().splitlines()]
    seatings = {r["seating_index"] for r in rows if r["type"] == "seating"}
    assert seatings == set(range(24))
    dec = [r for r in rows if r["type"] == "decision"]
    assert {r["seating_index"] for r in dec} == set(range(24))
    for r in dec:
        assert len(r["unseen_by_rank"]) == 13 and 0 < sum(r["unseen_by_rank"]) <= 52
        own_up = sum(t != 52 for t in r["obs"][:6])
        assert own_up < 6 or r["stage"] == 1  # finished games (all up, nothing held) are dropped
        assert not (own_up == 6 and r["obs"][6] == 52)
    ends = [r for r in rows if r["type"] == "hole_end"]
    assert len(ends) == 24 * 2  # 1 traced game per seating x 2 holes


def test_rejects_shape_change(run, tmp_path):
    ck = torch.load(run / "base.pt", weights_only=False)
    ck["model_state_dict"]["mlp.4.bias"] = torch.zeros(17)
    torch.save(ck, tmp_path / "bad.pt")
    r = _pass(run, str(tmp_path / "bad.pt"))
    assert r.returncode != 0 and "shape" in r.stderr

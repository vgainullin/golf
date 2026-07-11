import json
from pathlib import Path

from src.tournament import (
    TournamentConfig,
    restore_tournament_config,
    save_tournament_config,
    tournament_config_dict,
)


def test_tournament_config_round_trip_restores_training_fields(tmp_path):
    original = TournamentConfig(
        generations=350,
        population_size=8,
        embedding_dim=64,
        hidden_dim_choices=[256],
        lr_range=(8.3e-5, 0.0024),
        reward_shaping="hindsight",
        win_bonus=10.0,
        loss_penalty=-5.0,
        output_dir=tmp_path,
        device="mps",
        wandb_run_name="original-run",
    )
    saved_path = save_tournament_config(original)

    resumed = TournamentConfig(
        generations=500,
        output_dir=tmp_path,
        resume=True,
        device="cpu",
        win_bonus=0.0,
        loss_penalty=0.0,
        wandb_run_name="continued-run",
    )
    assert restore_tournament_config(resumed)

    assert saved_path == tmp_path / "tournament_config.json"
    assert resumed.generations == 500
    assert resumed.output_dir == tmp_path
    assert resumed.resume is True
    assert resumed.device == "cpu"
    assert resumed.wandb_run_name == "continued-run"
    assert resumed.population_size == 8
    assert resumed.embedding_dim == 64
    assert resumed.hidden_dim_choices == [256]
    assert resumed.lr_range == (8.3e-5, 0.0024)
    assert resumed.reward_shaping == "hindsight"
    assert resumed.win_bonus == 10.0
    assert resumed.loss_penalty == -5.0


def test_tournament_config_omits_credentials_and_serializes_paths(tmp_path):
    config = TournamentConfig(
        output_dir=tmp_path,
        warmstart_checkpoint=Path("data/model.pt"),
        hf_token="secret-token",
    )

    data = tournament_config_dict(config)

    assert "hf_token" not in data
    assert data["output_dir"] == str(tmp_path)
    assert data["warmstart_checkpoint"] == "data/model.pt"

    path = save_tournament_config(config)
    assert json.loads(path.read_text()) == data


def test_restore_missing_config_fails_closed(tmp_path):
    config = TournamentConfig(
        output_dir=tmp_path,
        resume=True,
        win_bonus=3.0,
        loss_penalty=-1.0,
    )

    try:
        restore_tournament_config(config)
    except RuntimeError as exc:
        assert "--allow-legacy-resume" in str(exc)
    else:
        raise AssertionError("legacy resume should require an explicit override")


def test_legacy_resume_override_keeps_explicit_values(tmp_path, capsys):
    config = TournamentConfig(
        output_dir=tmp_path,
        resume=True,
        allow_legacy_resume=True,
        win_bonus=3.0,
        loss_penalty=-1.0,
    )

    assert not restore_tournament_config(config)
    assert config.win_bonus == 3.0
    assert config.loss_penalty == -1.0
    assert "explicit legacy override" in capsys.readouterr().out

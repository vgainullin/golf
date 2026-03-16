"""Multi-objective hyperparameter search for tournament training using Optuna."""

import argparse
import json
import traceback
from pathlib import Path
from typing import Tuple

import optuna

from .tournament import TournamentConfig, TournamentTrainer


_QUICK = False
_TRIAL_DIR = "data/optuna_trials"
_STUDY_NAME = ""


def _parse_results(trial: optuna.Trial, output_dir: Path, generations: int) -> Tuple[float, float]:
    """Parse metrics_log.jsonl and return (solo_final, solo_mid)."""
    mid_gen = max(1, generations // 2)

    metrics_path = output_dir / "metrics_log.jsonl"
    if not metrics_path.exists():
        trial.set_user_attr("error", "metrics_log.jsonl not found")
        return (float("inf"), float("inf"))

    solo_by_gen = {}
    with open(metrics_path) as f:
        for line in f:
            m = json.loads(line)
            gen = m.get("generation")
            solo = m.get("eval/best_solo")
            if gen is not None and solo is not None:
                solo_by_gen[gen] = solo

    solo_mid = solo_by_gen.get(mid_gen, float("inf"))
    solo_final = solo_by_gen.get(generations, float("inf"))

    trial.set_user_attr("solo_mid", solo_mid)
    trial.set_user_attr("solo_final", solo_final)
    trial.set_user_attr("generations", generations)
    trial.set_user_attr("mid_gen", mid_gen)

    return (solo_final, solo_mid)


def _run_trial(trial: optuna.Trial, config: TournamentConfig, generations: int) -> Tuple[float, float]:
    """Run a tournament and return parsed results."""
    try:
        TournamentTrainer(config).run()
    except Exception:
        tb = traceback.format_exc()
        trial.set_user_attr("error", tb)
        print(f"Trial {trial.number} failed:\n{tb}")
        return (float("inf"), float("inf"))

    return _parse_results(trial, config.output_dir, generations)


def objective(trial: optuna.Trial) -> Tuple[float, float]:
    """Run one tournament trial and return (solo_gen_final, solo_gen_mid)."""
    # Sample hyperparameters
    epsilon_start = trial.suggest_float("epsilon_start", 0.3, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.15)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    lr_low = trial.suggest_float("lr_low", 1e-5, 1e-3, log=True)
    lr_high = trial.suggest_float("lr_high", 1e-3, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    buffer_capacity = trial.suggest_categorical("buffer_capacity", [50_000, 100_000, 200_000])
    updates_per_episode = trial.suggest_int("updates_per_episode", 2, 8)
    target_update_interval = trial.suggest_int("target_update_interval", 200, 1000)
    episodes_per_gen = trial.suggest_categorical("episodes_per_gen", [250, 500, 1000])
    tau = trial.suggest_float("tau", 0.0, 0.01)
    mutation_rate = trial.suggest_float("mutation_rate", 0.1, 0.5)
    mutation_sigma = trial.suggest_float("mutation_sigma", 0.1, 0.4)

    if _QUICK:
        generations = 4
        population_size = 4
        eval_games = 10
        episodes_per_gen = 50
        buffer_capacity = 10_000
    else:
        generations = 10
        population_size = 12
        eval_games = 100

    output_dir = Path(f"{_TRIAL_DIR}/{_STUDY_NAME}/trial_{trial.number}")

    model_variant = trial.suggest_categorical("model_variant", ["v2s", "v3"])

    config = TournamentConfig(
        generations=generations,
        population_size=population_size,
        eval_games_per_matchup=eval_games,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        gamma=gamma,
        lr_range=(lr_low, lr_high),
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        updates_per_episode=updates_per_episode,
        target_update_interval=target_update_interval,
        episodes_per_gen=episodes_per_gen,
        tau=tau,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        reward_shaping="hindsight",
        model_variant=model_variant,
        output_dir=output_dir,
        seed=trial.number,
    )

    return _run_trial(trial, config, generations)


def objective_v3_explore(trial: optuna.Trial) -> Tuple[float, float]:
    """Round 4: informed by r2 results.

    r2 best (12.55): buf=50k, lr_high=4e-3, eps/gen=1500, batch=512, updates=7.
    Tightened ranges around what worked. Added hidden_dim search.
    """
    epsilon_start = trial.suggest_float("epsilon_start", 0.75, 0.95)
    epsilon_end = trial.suggest_float("epsilon_end", 0.03, 0.08)
    lr_low = trial.suggest_float("lr_low", 5e-5, 3e-4, log=True)
    lr_high = trial.suggest_float("lr_high", 2e-3, 5e-3, log=True)
    updates_per_episode = trial.suggest_int("updates_per_episode", 5, 9)
    target_update_interval = trial.suggest_int("target_update_interval", 600, 1000)
    buffer_capacity = trial.suggest_categorical("buffer_capacity", [50_000, 75_000, 100_000])
    episodes_per_gen = trial.suggest_categorical("episodes_per_gen", [1000, 1500, 2000])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    if _QUICK:
        generations = 4
        population_size = 1
        eval_games = 10
        solo_eval = 50
        episodes_per_gen = 50
        buffer_capacity = 10_000
    else:
        generations = 6
        population_size = 1
        eval_games = 10
        solo_eval = 100

    output_dir = Path(f"{_TRIAL_DIR}/{_STUDY_NAME}/trial_{trial.number}")

    config = TournamentConfig(
        generations=generations,
        population_size=population_size,
        eval_games_per_matchup=eval_games,
        solo_eval_games=solo_eval,
        max_train_rounds=1,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        gamma=0.99,
        lr_range=(lr_low, lr_high),
        batch_size=trial.suggest_categorical("batch_size", [256, 512]),
        buffer_capacity=buffer_capacity,
        updates_per_episode=updates_per_episode,
        target_update_interval=target_update_interval,
        episodes_per_gen=episodes_per_gen,
        tau=0.0,
        mutation_rate=0.3,
        mutation_sigma=0.2,
        reward_shaping="hindsight",
        model_variant="v3",
        embedding_dim=64,
        hidden_dim_choices=[hidden_dim],
        output_dir=output_dir,
        seed=trial.number,
    )

    return _run_trial(trial, config, generations)


def _format_trial(t: optuna.trial.FrozenTrial) -> str:
    """Format a single trial with all params."""
    p = t.params
    lines = [f"  Trial {t.number}: final={t.values[0]:.2f}  mid={t.values[1]:.2f}"]
    for k, v in sorted(p.items()):
        if isinstance(v, float):
            lines.append(f"    {k}: {v:.4g}")
        else:
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def print_pareto_front(study: optuna.Study) -> None:
    """Print Pareto front and all completed trials with full params."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pareto = study.best_trials

    if not completed:
        print("No completed trials yet.")
        return

    print(f"\n=== Pareto front ({len(pareto)} trials) ===")
    for t in sorted(pareto, key=lambda t: t.values[0]):
        print(_format_trial(t))

    print(f"\n=== All completed trials ({len(completed)}) ===")
    for t in sorted(completed, key=lambda t: t.values[0]):
        print(_format_trial(t))


_MODE_DEFAULTS = {
    "hpo": {"study_name": "golf-tournament-hpo", "objective": objective},
    "v3-explore": {"study_name": "v3-explore", "objective": objective_v3_explore},
}


def main():
    p = argparse.ArgumentParser(description="Optuna HPO for tournament training")
    p.add_argument("--mode", choices=list(_MODE_DEFAULTS.keys()), default="hpo",
                   help="Search mode: 'hpo' (general) or 'v3-explore' (exploration-focused)")
    p.add_argument("--n-trials", type=int, default=20, help="Number of trials to run")
    p.add_argument("--show-results", action="store_true", help="Print Pareto front and exit")
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: 4 gens, 4 agents, 50 eps, 10 eval games")
    p.add_argument("--db", type=str, default="sqlite:///data/optuna_study.db",
                   help="Optuna storage URL")
    p.add_argument("--study-name", type=str, default=None,
                   help="Study name (default: per-mode)")
    args = p.parse_args()

    global _QUICK, _STUDY_NAME
    _QUICK = args.quick

    mode_cfg = _MODE_DEFAULTS[args.mode]
    study_name = args.study_name or mode_cfg["study_name"]
    _STUDY_NAME = study_name
    obj_fn = mode_cfg["objective"]

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name=study_name,
        storage=args.db,
        load_if_exists=True,
    )

    if args.show_results:
        print_pareto_front(study)
        return

    study.optimize(obj_fn, n_trials=args.n_trials)
    print_pareto_front(study)


if __name__ == "__main__":
    main()

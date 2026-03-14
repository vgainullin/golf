"""Multi-objective hyperparameter search for tournament training using Optuna."""

import argparse
import json
import traceback
from pathlib import Path
from typing import Tuple

import optuna

from .tournament import TournamentConfig, TournamentTrainer


_QUICK = False


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
        # Override sampled values that dominate runtime
        episodes_per_gen = 50
        buffer_capacity = 10_000
    else:
        generations = 10
        population_size = 12
        eval_games = 100

    mid_gen = max(1, generations // 2)

    output_dir = Path(f"data/optuna_trials/trial_{trial.number}")

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

    try:
        TournamentTrainer(config).run()
    except Exception:
        tb = traceback.format_exc()
        trial.set_user_attr("error", tb)
        print(f"Trial {trial.number} failed:\n{tb}")
        return (float("inf"), float("inf"))

    # Parse metrics_log.jsonl for mid and final gen solo scores
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


def print_pareto_front(study: optuna.Study) -> None:
    """Print the Pareto-optimal trials."""
    trials = study.best_trials
    if not trials:
        print("No completed trials yet.")
        return

    print(f"\nPareto front ({len(trials)} trials):")
    print(f"{'Trial':>6}  {'Solo@10':>8}  {'Solo@5':>8}  Key params")
    print("-" * 70)
    for t in sorted(trials, key=lambda t: t.values[0]):
        params = t.params
        print(
            f"{t.number:>6}  {t.values[0]:>8.2f}  {t.values[1]:>8.2f}  "
            f"eps={params['epsilon_start']:.2f}->{params['epsilon_end']:.3f} "
            f"lr={params['lr_low']:.1e}-{params['lr_high']:.1e} "
            f"bs={params['batch_size']} buf={params['buffer_capacity']//1000}k"
        )

    print(f"\nAll completed trials: {len(study.trials)}")


def main():
    p = argparse.ArgumentParser(description="Optuna HPO for tournament training")
    p.add_argument("--n-trials", type=int, default=20, help="Number of trials to run")
    p.add_argument("--show-results", action="store_true", help="Print Pareto front and exit")
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: 4 gens, 4 agents, 50 eps, 10 eval games")
    p.add_argument("--db", type=str, default="sqlite:///data/optuna_study.db",
                   help="Optuna storage URL")
    p.add_argument("--study-name", type=str, default="golf-tournament-hpo")
    args = p.parse_args()

    global _QUICK
    _QUICK = args.quick

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
    )

    if args.show_results:
        print_pareto_front(study)
        return

    study.optimize(objective, n_trials=args.n_trials)
    print_pareto_front(study)


if __name__ == "__main__":
    main()

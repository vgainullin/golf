"""Multi-experiment training runner for DQN hyperparameter sweeps."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import itertools

from .dqn_offline import TrainingConfig, train_offline


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""
    name: str
    archive_prefix: Path
    base_output_dir: Path

    # Hyperparameters to sweep
    epochs: int = 20
    batch_size: int = 2048
    learning_rate: float = 1e-3
    gamma: float = 0.99
    target_update_interval: int = 750
    val_fraction: float = 0.1
    embedding_dim: int = 128
    hidden_dim: int = 256
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    max_steps: Optional[int] = None
    num_workers: int = 0


def create_experiment_grid(
    base_config: ExperimentConfig,
    param_grid: Dict[str, List[Any]],
) -> List[ExperimentConfig]:
    """Generate all combinations of hyperparameters from a grid.

    Args:
        base_config: Base experiment configuration
        param_grid: Dict mapping parameter names to lists of values to try

    Returns:
        List of ExperimentConfig objects, one per combination

    Example:
        param_grid = {
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "hidden_dim": [128, 256, 512],
            "batch_size": [1024, 2048],
        }
    """
    experiments = []

    # Get all parameter names and their value lists
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    # Generate all combinations
    for idx, combination in enumerate(itertools.product(*param_values)):
        # Create a copy of base config
        config_dict = asdict(base_config)

        # Update with this combination
        param_str_parts = []
        for param_name, value in zip(param_names, combination):
            config_dict[param_name] = value
            # Build a descriptive name
            param_str_parts.append(f"{param_name}={value}")

        # Create unique experiment name
        config_dict["name"] = f"exp_{idx:03d}_" + "_".join(param_str_parts)

        # Convert paths back from strings if needed
        config_dict["archive_prefix"] = Path(config_dict["archive_prefix"])
        config_dict["base_output_dir"] = Path(config_dict["base_output_dir"])

        experiments.append(ExperimentConfig(**config_dict))

    return experiments


def run_experiment(
    config: ExperimentConfig,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single training experiment.

    Args:
        config: Experiment configuration
        dry_run: If True, only print what would be done without training

    Returns:
        Dict with experiment results and metadata
    """
    output_dir = config.base_output_dir / config.name

    if dry_run:
        print(f"[DRY RUN] Would train: {config.name}")
        print(f"  Output: {output_dir}")
        print(f"  Config: {asdict(config)}")
        return {"status": "dry_run", "config": asdict(config)}

    print(f"\n{'='*80}")
    print(f"Starting experiment: {config.name}")
    print(f"{'='*80}")

    # Create TrainingConfig from ExperimentConfig
    train_config = TrainingConfig(
        archive_prefix=config.archive_prefix,
        output_dir=output_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        target_update_interval=config.target_update_interval,
        val_fraction=config.val_fraction,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        seed=config.seed,
        device=config.device,
        max_steps=config.max_steps,
        num_workers=config.num_workers,
    )

    try:
        # Run training
        result = train_offline(train_config)

        # Save experiment config alongside results
        config_path = output_dir / "experiment_config.json"
        with config_path.open("w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

        result["status"] = "success"
        result["config"] = asdict(config)

        # Extract final metrics
        history = result.get("history", [])
        if history:
            final_epoch = history[-1]
            result["final_train_loss"] = final_epoch.get("train_loss")
            result["final_val_loss"] = final_epoch.get("val_loss")
            result["final_steps"] = final_epoch.get("steps")

            # Find best validation loss
            val_losses = [epoch.get("val_loss", float("inf")) for epoch in history]
            best_val_loss = min(val_losses)
            best_epoch = val_losses.index(best_val_loss) + 1
            result["best_val_loss"] = best_val_loss
            result["best_epoch"] = best_epoch

        print(f"\n{'='*80}")
        print(f"Completed: {config.name}")
        print(f"  Final train loss: {result.get('final_train_loss', 'N/A'):.4f}")
        print(f"  Final val loss: {result.get('final_val_loss', 'N/A'):.4f}")
        print(f"  Best val loss: {result.get('best_val_loss', 'N/A'):.4f} (epoch {result.get('best_epoch', 'N/A')})")
        print(f"{'='*80}\n")

        return result

    except Exception as e:
        print(f"\n[ERROR] Experiment {config.name} failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "config": asdict(config),
        }


def run_experiment_suite(
    experiments: List[ExperimentConfig],
    dry_run: bool = False,
    results_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run a suite of experiments and save results.

    Args:
        experiments: List of experiment configurations to run
        dry_run: If True, only print what would be done
        results_file: Optional path to save aggregate results

    Returns:
        List of result dicts, one per experiment
    """
    print(f"\n{'#'*80}")
    print(f"EXPERIMENT SUITE: {len(experiments)} experiments")
    print(f"{'#'*80}\n")

    if dry_run:
        print("[DRY RUN MODE - No training will be performed]\n")

    results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {config.name}")
        result = run_experiment(config, dry_run=dry_run)
        results.append(result)

    # Save aggregate results
    if results_file and not dry_run:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved aggregate results to: {results_file}")

    # Print summary
    print(f"\n{'#'*80}")
    print("EXPERIMENT SUITE SUMMARY")
    print(f"{'#'*80}")

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    print(f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}")

    if successful:
        print(f"\nTop 5 by best validation loss:")
        sorted_results = sorted(
            successful,
            key=lambda r: r.get("best_val_loss", float("inf"))
        )
        for i, result in enumerate(sorted_results[:5], 1):
            config_name = result.get("config", {}).get("name", "unknown")
            best_val = result.get("best_val_loss", float("inf"))
            print(f"  {i}. {config_name}: {best_val:.4f}")

    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DQN training experiments with hyperparameter sweeps."
    )
    parser.add_argument(
        "--archive-prefix",
        type=Path,
        default=Path("tmp/tensor_logs_batch/tensor_transitions_combined"),
        help="Path prefix to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/experiments"),
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file with experiment grid configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running them",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Default number of epochs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers",
    )
    return parser.parse_args(argv)


def load_experiment_grid(config_file: Path) -> Dict[str, List[Any]]:
    """Load experiment grid from JSON file.

    Expected format:
    {
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "hidden_dim": [128, 256, 512],
        "batch_size": [1024, 2048]
    }
    """
    with config_file.open() as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Create base config
    base_config = ExperimentConfig(
        name="base",
        archive_prefix=args.archive_prefix,
        base_output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
    )

    # Load or create experiment grid
    if args.config:
        print(f"Loading experiment grid from: {args.config}")
        param_grid = load_experiment_grid(args.config)
    else:
        # Default grid for quick testing
        print("Using default experiment grid")
        param_grid = {
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "hidden_dim": [128, 256],
            "batch_size": [1024, 2048],
        }

    # Generate experiments
    experiments = create_experiment_grid(base_config, param_grid)

    print(f"\nGenerated {len(experiments)} experiments from parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Run experiments
    results_file = args.output_dir / "experiment_results.json"
    results = run_experiment_suite(
        experiments,
        dry_run=args.dry_run,
        results_file=results_file,
    )

    print(f"\nAll experiments complete!")


if __name__ == "__main__":
    main()

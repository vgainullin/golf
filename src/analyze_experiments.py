"""Visualization and analysis tools for training experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (must be set before importing pyplot)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_experiment_results(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Load all experiment results from a directory.

    Args:
        experiments_dir: Directory containing experiment subdirectories

    Returns:
        List of experiment data dicts
    """
    experiments = []

    for subdir in sorted(experiments_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Load training history
        history_file = subdir / "training_history.json"
        if not history_file.exists():
            continue

        with history_file.open() as f:
            history = json.load(f)

        # Load experiment config
        config_file = subdir / "experiment_config.json"
        config = {}
        if config_file.exists():
            with config_file.open() as f:
                config = json.load(f)

        experiments.append({
            "name": subdir.name,
            "path": subdir,
            "history": history,
            "config": config,
        })

    return experiments


def create_training_curves(
    experiments: List[Dict[str, Any]],
    output_dir: Path,
    max_plots: int = 20,
) -> None:
    """Create training curve plots for experiments.

    Args:
        experiments: List of experiment data
        output_dir: Directory to save plots
        max_plots: Maximum number of experiments to plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot all training curves on one figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, exp in enumerate(experiments[:max_plots]):
        history = exp["history"]
        if not history:
            continue

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h.get("val_loss", float("nan")) for h in history]

        label = exp["name"][:40]  # Truncate long names

        ax1.plot(epochs, train_losses, label=label, alpha=0.7)
        ax2.plot(epochs, val_losses, label=label, alpha=0.7)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss Curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Curves")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to: {output_dir / 'training_curves_all.png'}")


def create_hyperparameter_analysis(
    experiments: List[Dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Analyze the relationship between hyperparameters and performance.

    Args:
        experiments: List of experiment data
        output_dir: Directory to save analysis

    Returns:
        DataFrame with hyperparameter analysis
    """
    rows = []

    for exp in experiments:
        history = exp["history"]
        config = exp["config"]

        if not history:
            continue

        # Extract final and best metrics
        val_losses = [h.get("val_loss", float("inf")) for h in history]
        best_val_loss = min(val_losses)
        final_val_loss = val_losses[-1] if val_losses else float("nan")

        train_losses = [h["train_loss"] for h in history]
        final_train_loss = train_losses[-1] if train_losses else float("nan")

        # Extract hyperparameters
        row = {
            "experiment": exp["name"],
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "final_train_loss": final_train_loss,
            "epochs_trained": len(history),
            "learning_rate": config.get("learning_rate", float("nan")),
            "batch_size": config.get("batch_size", float("nan")),
            "hidden_dim": config.get("hidden_dim", float("nan")),
            "embedding_dim": config.get("embedding_dim", float("nan")),
            "gamma": config.get("gamma", float("nan")),
            "weight_decay": config.get("weight_decay", float("nan")),
            "grad_clip": config.get("grad_clip", float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("No experiment data to analyze")
        return df

    # Sort by best validation loss
    df = df.sort_values("best_val_loss")

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hyperparameter_analysis.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    print(f"\nSaved hyperparameter analysis to: {csv_path}")

    return df


def create_hyperparameter_heatmaps(
    experiments: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Create heatmaps showing hyperparameter vs performance.

    Args:
        experiments: List of experiment data
        output_dir: Directory to save plots
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping heatmaps")
        return

    df = create_hyperparameter_analysis(experiments, output_dir)

    if len(df) == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create 2D heatmaps for common hyperparameter pairs
    heatmap_pairs = [
        ("learning_rate", "hidden_dim"),
        ("learning_rate", "batch_size"),
        ("batch_size", "hidden_dim"),
    ]

    for param1, param2 in heatmap_pairs:
        if param1 not in df.columns or param2 not in df.columns:
            continue

        # Group by both parameters and take the best val loss
        pivot_data = df.groupby([param1, param2])["best_val_loss"].min().reset_index()

        if len(pivot_data) < 2:
            continue

        # Create pivot table
        pivot = pivot_data.pivot(index=param2, columns=param1, values="best_val_loss")

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"{x:.2g}" for x in pivot.columns])
        ax.set_yticklabels([f"{y:.2g}" for y in pivot.index])

        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f"Best Validation Loss vs {param1} and {param2}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Best Validation Loss")

        # Annotate cells with values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f"{val:.3f}",
                                   ha="center", va="center", color="white", fontsize=9)

        plt.tight_layout()
        filename = f"heatmap_{param1}_vs_{param2}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved heatmap to: {output_dir / filename}")


def print_summary_statistics(experiments: List[Dict[str, Any]]) -> None:
    """Print summary statistics across all experiments.

    Args:
        experiments: List of experiment data
    """
    if not experiments:
        print("No experiments to analyze")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Collect metrics
    best_val_losses = []
    final_train_losses = []
    epochs_list = []

    for exp in experiments:
        history = exp["history"]
        if not history:
            continue

        val_losses = [h.get("val_loss", float("inf")) for h in history]
        train_losses = [h["train_loss"] for h in history]

        best_val_losses.append(min(val_losses))
        final_train_losses.append(train_losses[-1] if train_losses else float("nan"))
        epochs_list.append(len(history))

    if best_val_losses:
        print(f"\nTotal experiments: {len(experiments)}")
        print(f"Experiments with training data: {len(best_val_losses)}")
        print(f"\nBest Validation Loss:")
        print(f"  Min: {np.min(best_val_losses):.4f}")
        print(f"  Max: {np.max(best_val_losses):.4f}")
        print(f"  Mean: {np.mean(best_val_losses):.4f}")
        print(f"  Median: {np.median(best_val_losses):.4f}")
        print(f"  Std: {np.std(best_val_losses):.4f}")

        print(f"\nFinal Training Loss:")
        print(f"  Min: {np.nanmin(final_train_losses):.4f}")
        print(f"  Max: {np.nanmax(final_train_losses):.4f}")
        print(f"  Mean: {np.nanmean(final_train_losses):.4f}")
        print(f"  Median: {np.nanmedian(final_train_losses):.4f}")

        print(f"\nEpochs Trained:")
        print(f"  Min: {np.min(epochs_list)}")
        print(f"  Max: {np.max(epochs_list)}")
        print(f"  Mean: {np.mean(epochs_list):.1f}")

        # Top 5 experiments by best val loss
        experiments_with_loss = [
            (exp, min([h.get("val_loss", float("inf")) for h in exp["history"]]))
            for exp in experiments if exp["history"]
        ]
        experiments_with_loss.sort(key=lambda x: x[1])

        print(f"\n{'='*80}")
        print("TOP 5 EXPERIMENTS BY VALIDATION LOSS")
        print(f"{'='*80}")

        for i, (exp, val_loss) in enumerate(experiments_with_loss[:5], 1):
            config = exp["config"]
            print(f"\n{i}. {exp['name']}")
            print(f"   Best val loss: {val_loss:.4f}")
            print(f"   Learning rate: {config.get('learning_rate', 'N/A')}")
            print(f"   Hidden dim: {config.get('hidden_dim', 'N/A')}")
            print(f"   Batch size: {config.get('batch_size', 'N/A')}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize training experiment results."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("tmp/experiments"),
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/analysis"),
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=20,
        help="Maximum number of experiments to include in plots",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if not args.experiments_dir.exists():
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        return

    print(f"Loading experiments from: {args.experiments_dir}")
    experiments = load_experiment_results(args.experiments_dir)

    if not experiments:
        print("No experiments found")
        return

    print(f"Found {len(experiments)} experiments")

    # Print summary statistics
    print_summary_statistics(experiments)

    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    create_training_curves(experiments, args.output_dir, max_plots=args.max_plots)
    create_hyperparameter_analysis(experiments, args.output_dir)
    create_hyperparameter_heatmaps(experiments, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

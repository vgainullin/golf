"""Generate SVG plots for offline DQN training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dqn_offline import (
    GolfDQN,
    OfflineTransitionDataset,
    TrainingConfig,
    build_transition_arrays,
)


def render_svg_line_plot(
    *,
    output_path: Path,
    x_values: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    x_label: str,
    y_label: str,
    x_ticks: int = 6,
    y_ticks: int = 6,
) -> None:
    if not x_values:
        raise ValueError("x_values must not be empty.")
    if any(len(values) != len(x_values) for _, values, _ in series):
        raise ValueError("Each series must align with x_values.")

    width, height = 900, 540
    margin_left, margin_bottom, margin_top, margin_right = 80, 70, 70, 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    all_y = [float(v) for _, values, _ in series for v in values if not np.isnan(v)]
    if not all_y:
        raise ValueError("No numeric values to plot.")
    y_min = float(min(all_y))
    y_max = float(max(all_y))
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    def x_coord(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        norm = (value - y_min) / (y_max - y_min)
        return margin_top + (1 - norm) * plot_height

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; }</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" '
        'stroke="#333" stroke-width="2" />',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" '
        f'y2="{height - margin_bottom}" stroke="#333" stroke-width="2" />',
    ]

    def axis_ticks(count: int) -> Iterable[float]:
        step = (1 / max(count - 1, 1))
        for i in range(count):
            yield i * step

    # X-axis ticks
    for frac in axis_ticks(x_ticks):
        value = x_min + frac * (x_max - x_min)
        x = x_coord(value)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom:.2f}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom + 6:.2f}" stroke="#555" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 25:.2f}" font-size="15" '
            f'text-anchor="middle" fill="#444">{value:.2f}</text>'
        )

    # Y-axis ticks
    for frac in axis_ticks(y_ticks):
        value = y_min + frac * (y_max - y_min)
        y = y_coord(value)
        svg_parts.append(
            f'<line x1="{margin_left - 6:.2f}" y1="{y:.2f}" x2="{margin_left:.2f}" '
            f'y2="{y:.2f}" stroke="#555" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10:.2f}" y="{y + 5:.2f}" font-size="15" '
            f'text-anchor="end" fill="#444">{value:.3f}</text>'
        )
        svg_parts.append(
            f'<line x1="{margin_left:.2f}" y1="{y:.2f}" x2="{width - margin_right:.2f}" '
            f'y2="{y:.2f}" stroke="#e0e0e0" stroke-width="1" />'
        )

    # Series polylines
    for label, values, color in series:
        points = " ".join(
            f"{x_coord(x_values[i]):.2f},{y_coord(values[i]):.2f}" for i in range(len(x_values))
        )
        svg_parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />'
        )

    # Legend
    legend_width = 200
    legend_height = 30 * len(series) + 20
    legend_x = width - margin_right - legend_width
    legend_y = margin_top + 10
    svg_parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" '
        'fill="#fff" stroke="#ccc" stroke-width="1" rx="8" ry="8" />'
    )

    for idx, (label, _, color) in enumerate(series):
        y = legend_y + 25 + idx * 30
        svg_parts.append(
            f'<line x1="{legend_x + 15}" y1="{y}" x2="{legend_x + 55}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" />'
        )
        svg_parts.append(
            f'<text x="{legend_x + 65}" y="{y + 5}" font-size="16" fill="#333">{label}</text>'
        )

    svg_parts.append(
        f'<text x="{width / 2}" y="{margin_top - 25}" font-size="22" text-anchor="middle" '
        f'fill="#222">{title}</text>'
    )
    svg_parts.append(
        f'<text x="{width / 2}" y="{height - 20}" font-size="16" text-anchor="middle" '
        f'fill="#333">{x_label}</text>'
    )
    svg_parts.append(
        f'<text transform="translate(30 {height / 2}) rotate(-90)" font-size="16" '
        f'text-anchor="middle" fill="#333">{y_label}</text>'
    )
    svg_parts.append("</svg>")

    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def load_history(history_path: Path) -> List[dict]:
    with history_path.open() as fh:
        history = json.load(fh)
    if not isinstance(history, list):
        raise ValueError("Training history JSON must contain a list of epochs.")
    return history


def create_loss_plot(history: List[dict], output_dir: Path) -> Path:
    epochs = [float(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]
    output_path = output_dir / "training_loss.svg"
    render_svg_line_plot(
        output_path=output_path,
        x_values=epochs,
        series=[
            ("Train Loss", train_loss, "#1f77b4"),
            ("Validation Loss", val_loss, "#ff7f0e"),
        ],
        title="Offline DQN Training History",
        x_label="Epoch",
        y_label="Smooth L1 Loss",
    )
    return output_path


def load_config_from_checkpoint(checkpoint_path: Path) -> TrainingConfig:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = state.get("config")
    if config_dict is None:
        raise ValueError("Checkpoint is missing TrainingConfig data.")
    config_kwargs = dict(config_dict)
    # Ensure Path fields are restored.
    config_kwargs["archive_prefix"] = Path(config_kwargs["archive_prefix"])
    config_kwargs["output_dir"] = Path(config_kwargs["output_dir"])
    return TrainingConfig(**config_kwargs)


def create_q_value_plot(
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 4096,
    quantile_resolution: int = 20,
) -> Tuple[Path, dict]:
    config = load_config_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transitions = build_transition_arrays(config.archive_prefix)
    dataset = OfflineTransitionDataset(transitions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = GolfDQN(config.embedding_dim, config.hidden_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    max_q_vals: List[torch.Tensor] = []
    action_q_vals: List[torch.Tensor] = []

    with torch.no_grad():
        for states, actions, _, _, _, stages, _ in loader:
            states = states.to(device)
            actions = actions.to(device)
            stages = stages.to(device)
            q_values = model(states, stages)
            max_q_vals.append(q_values.max(dim=1).values.cpu())
            action_q_vals.append(q_values.gather(1, actions.unsqueeze(1)).squeeze(1).cpu())

    max_q = torch.cat(max_q_vals).numpy()
    action_q = torch.cat(action_q_vals).numpy()

    quantiles = np.linspace(0.0, 1.0, quantile_resolution + 1)
    max_q_quantiles = np.quantile(max_q, quantiles)
    action_q_quantiles = np.quantile(action_q, quantiles)

    output_path = output_dir / "q_value_quantiles.svg"
    render_svg_line_plot(
        output_path=output_path,
        x_values=(quantiles * 100).tolist(),
        series=[
            ("Max Q (per state)", max_q_quantiles.tolist(), "#2ca02c"),
            ("Q(action)", action_q_quantiles.tolist(), "#d62728"),
        ],
        title="Predicted Q-Value Distribution",
        x_label="Quantile (%)",
        y_label="Q-Value",
    )

    stats = {
        "max_q_mean": float(np.mean(max_q)),
        "max_q_std": float(np.std(max_q)),
        "max_q_min": float(np.min(max_q)),
        "max_q_max": float(np.max(max_q)),
        "action_q_mean": float(np.mean(action_q)),
        "action_q_std": float(np.std(action_q)),
        "action_q_min": float(np.min(action_q)),
        "action_q_max": float(np.max(action_q)),
        "sample_count": int(len(max_q)),
    }
    stats_path = output_dir / "q_value_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return output_path, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot offline DQN training history and Q-value stats."
    )
    parser.add_argument(
        "--history",
        default="tmp/offline_dqn_attempt/training_history.json",
        help="Path to training_history.json file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="tmp/offline_dqn_attempt/offline_dqn.pt",
        help="Path to offline DQN checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/offline_dqn_attempt",
        help="Directory to write SVG plots and stats.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for Q-value statistics evaluation.",
    )
    parser.add_argument(
        "--quantile-resolution",
        type=int,
        default=20,
        help="Number of quantile segments for Q-value plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)
    loss_plot = create_loss_plot(history, output_dir)
    q_plot, stats = create_q_value_plot(
        checkpoint_path,
        output_dir,
        batch_size=args.batch_size,
        quantile_resolution=args.quantile_resolution,
    )

    summary_path = output_dir / "q_value_stats.json"
    print(f"Created loss plot: {loss_plot}")
    print(f"Created Q-value plot: {q_plot}")
    print(f"Q-value statistics saved to: {summary_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

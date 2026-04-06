# DQN Training Pipeline

This directory contains configuration files and documentation for the improved DQN training pipeline.

## Overview

The training pipeline consists of three main stages:

1. **Training Experiments**: Train multiple DQN agents with different hyperparameters
2. **Evaluation**: Test trained agents against baseline players (Random, Heuristic)
3. **Analysis**: Compare results and visualize training curves

## Quick Start

### 1. Run Training Experiments

Train multiple agents with a hyperparameter sweep:

```bash
# Quick sweep (12 experiments: 3 LRs × 2 hidden dims × 2 batch sizes)
python -m src.experiment_runner \
  --config configs/experiment_quick.json \
  --output-dir tmp/experiments_quick \
  --epochs 20 \
  --device cuda

# Comprehensive sweep (hundreds of experiments)
python -m src.experiment_runner \
  --config configs/experiment_comprehensive.json \
  --output-dir tmp/experiments_full \
  --epochs 30 \
  --device cuda
```

**Dry run** to see what would be executed:
```bash
python -m src.experiment_runner \
  --config configs/experiment_quick.json \
  --dry-run
```

### 2. Evaluate Trained Agents

> **Note:** `src/evaluate_agents.py` is deprecated (old non-vectorized loop).
> Use the vectorized evaluation scripts below instead.

```bash
# Evaluate Hall-of-Fame checkpoint [DQN, R, H, R]
uv run python -m scripts.eval_hof --repo-id vgainullin/golf --games 1000 --holes 9

# Evaluate all tournament agents vs 3 random opponents (batched GPU)
uv run python -m scripts.eval_vs_random --tournament-dir data/exp11_cyclic --games 200 --holes 9

# Benchmark heuristic baselines
uv run python -m scripts.eval_heuristics --games 5000 --holes 9
```

### 3. Analyze Results

Generate visualizations and comparison tables:

```bash
python -m src.analyze_experiments \
  --experiments-dir tmp/experiments_quick \
  --output-dir tmp/analysis_quick
```

This generates:
- Training loss curves for all experiments
- Hyperparameter heatmaps
- Summary statistics
- Top performer rankings

## Configuration Files

### `experiment_quick.json`
A quick hyperparameter sweep suitable for testing:
- 3 learning rates: [1e-4, 3e-4, 1e-3]
- 2 hidden dimensions: [128, 256]
- 2 batch sizes: [1024, 2048]
- **Total**: 12 experiments

### `experiment_comprehensive.json`
A comprehensive sweep for finding optimal hyperparameters:
- 4 learning rates
- 3 hidden dimensions
- 3 embedding dimensions
- 4 batch sizes
- 3 weight decay values
- 2 gamma values
- **Total**: 864 experiments (will take significant time!)

## Custom Configuration

Create your own hyperparameter sweep by creating a JSON file:

```json
{
  "learning_rate": [1e-4, 3e-4, 1e-3],
  "hidden_dim": [128, 256, 512],
  "batch_size": [1024, 2048],
  "weight_decay": [0.0, 1e-5],
  "gamma": [0.95, 0.99]
}
```

Any parameter from `TrainingConfig` can be swept.

## Advanced Options

### Early Stopping

Add early stopping to prevent overfitting:

```bash
python -m src.dqn_offline \
  --archive-prefix tmp/tensor_logs_batch/tensor_transitions_combined \
  --output-dir tmp/my_training \
  --epochs 50 \
  --early-stopping-patience 5 \
  --early-stopping-min-delta 0.001
```

This will stop training if validation loss doesn't improve by at least 0.001 for 5 consecutive epochs.

### Training from Scratch

If you need to generate new training data:

```bash
# Generate tensor transition logs
python -m src.simulation \
  --games 10000 \
  --holes 9 \
  --log-tensors \
  --tensor-log-dir data/training_logs \
  --workers 4

# Train on the new data
python -m src.dqn_offline \
  --archive-prefix data/training_logs/tensor_transitions_combined \
  --output-dir tmp/my_model \
  --epochs 20
```

## Interpreting Results

### Training Metrics

- **Train Loss**: Lower is better. Should decrease over epochs.
- **Val Loss**: Lower is better. This is what you optimize for.
- **Best Val Loss**: The lowest validation loss achieved during training.

### Evaluation Metrics

- **Win Rate**: Percentage of games won (higher is better)
- **Score**: Average score per game (LOWER is better in golf!)
- **Score Delta**: Agent score - opponent score (negative is good!)
- **Rank**: Average finishing position (1.0 = always wins)

### Good Performance Indicators

1. **Win rate > 25%** (baseline is 25% if all players were equal)
2. **Score delta < 0** (agent scores lower than opponents)
3. **Training converges** (validation loss stabilizes)
4. **No overfitting** (train loss and val loss stay close)

## Workflow Example

Full workflow to find the best agent:

```bash
# 1. Train multiple experiments
python -m src.experiment_runner \
  --config configs/experiment_quick.json \
  --output-dir tmp/exp_run1 \
  --epochs 20

# 2. Analyze training results
python -m src.analyze_experiments \
  --experiments-dir tmp/exp_run1 \
  --output-dir tmp/analysis_run1

# Review tmp/analysis_run1/hyperparameter_analysis.csv
# Identify promising hyperparameter ranges

# 3. Evaluate top performers
uv run python -m scripts.eval_vs_random \
  --tournament-dir tmp/exp_run1 \
  --games 200 --holes 9

# 4. Refine hyperparameters around best performers
# Edit a new config file with tighter ranges
# Repeat steps 1-3
```

## Parallel Execution

For faster experimentation:

1. **Data loading**: Use `--num-workers N` for parallel data loading during training
2. **Multiple GPUs**: Run different experiments on different GPUs manually
3. **Batch evaluation**: The evaluation script automatically processes all experiments

## Output Directory Structure

```
tmp/experiments_quick/
├── experiment_results.json          # Aggregate results
├── exp_000_learning_rate=0.0001_hidden_dim=128_batch_size=1024/
│   ├── offline_dqn.pt              # Model checkpoint
│   ├── training_history.json       # Loss curves
│   └── experiment_config.json      # Hyperparameters
├── exp_001_learning_rate=0.0001_hidden_dim=128_batch_size=2048/
│   └── ...
└── ...

tmp/evaluations_quick/
├── evaluation_results.json          # All evaluation data
├── agent_comparison.csv             # Comparison table
├── exp_000_..._games.csv           # Per-game results
├── exp_000_..._summary.json        # Evaluation metrics
└── ...

tmp/analysis_quick/
├── training_curves_all.png          # Loss curves plot
├── hyperparameter_analysis.csv      # Performance vs hyperparams
├── heatmap_learning_rate_vs_hidden_dim.png
└── ...
```

## Tips

1. **Start small**: Use `experiment_quick.json` first to verify everything works
2. **Monitor GPU usage**: Use `nvidia-smi` to ensure efficient GPU utilization
3. **Check early**: Look at training curves after a few experiments to catch issues
4. **Iterate**: Use insights from analysis to refine hyperparameter ranges
5. **Compare fairly**: Always use the same evaluation settings (games, holes, seed)

## Troubleshooting

### Training is too slow
- Reduce number of epochs
- Increase batch size
- Use `--num-workers` for data loading
- Use GPU with `--device cuda`

### Models aren't learning
- Check training data quality (`tmp/tensor_logs_batch/`)
- Try different learning rates (wider range)
- Increase model capacity (hidden_dim, embedding_dim)
- Check for bugs in reward signal

### Evaluation shows poor performance
- Ensure enough training data was collected
- Try longer training (more epochs)
- Check if validation loss was decreasing
- Compare to baseline training loss from previous runs

## Next Steps

After finding a good agent:

1. **Test on more games**: Increase `--games` to 500-1000 for statistical significance
2. **Iterate on architecture**: Try different model architectures in `dqn_offline.py`
3. **Collect more data**: Generate more training data with different player strategies
4. **Online fine-tuning**: Implement online RL to further improve the agent

## References

- Training code: `src/dqn_offline.py`
- Experiment runner: `src/experiment_runner.py`
- Evaluation: `scripts/eval_hof.py`, `scripts/eval_vs_random.py`, `scripts/eval_heuristics.py`
- Analysis: `src/analyze_experiments.py`

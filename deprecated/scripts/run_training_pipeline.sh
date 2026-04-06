#!/bin/bash
# Quick reference script for the DQN training pipeline
# This is a reference - customize paths and parameters as needed

set -e  # Exit on error

# Configuration
DATA_PREFIX="tmp/tensor_logs_batch/tensor_transitions_combined"
EXPERIMENTS_DIR="tmp/experiments_$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="tmp/evaluations_$(date +%Y%m%d_%H%M%S)"
ANALYSIS_DIR="tmp/analysis_$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="configs/experiment_quick.json"
EPOCHS=20
EVAL_GAMES=100
EVAL_HOLES=9
DEVICE="auto"  # auto, cuda, or cpu

echo "================================================================"
echo "DQN Training Pipeline"
echo "================================================================"
echo "Experiments: $EXPERIMENTS_DIR"
echo "Evaluations: $EVAL_DIR"
echo "Analysis: $ANALYSIS_DIR"
echo "================================================================"
echo ""

# Step 1: Train experiments
echo "[1/3] Training experiments..."
python -m src.experiment_runner \
  --archive-prefix "$DATA_PREFIX" \
  --config "$CONFIG_FILE" \
  --output-dir "$EXPERIMENTS_DIR" \
  --epochs $EPOCHS \
  --device $DEVICE

echo ""
echo "✓ Training complete!"
echo ""

# Step 2: Evaluate agents
echo "[2/3] Evaluating agents..."
# NOTE: src.evaluate_agents is deprecated. Use the vectorized eval scripts:
#   uv run python -m scripts.eval_vs_random --tournament-dir "$EXPERIMENTS_DIR" --games $EVAL_GAMES --holes $EVAL_HOLES
#   uv run python -m scripts.eval_hof --games $EVAL_GAMES --holes $EVAL_HOLES
uv run python -m scripts.eval_vs_random \
  --tournament-dir "$EXPERIMENTS_DIR" \
  --games $EVAL_GAMES \
  --holes $EVAL_HOLES

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 3: Analyze results
echo "[3/3] Analyzing results..."
python -m src.analyze_experiments \
  --experiments-dir "$EXPERIMENTS_DIR" \
  --output-dir "$ANALYSIS_DIR"

echo ""
echo "✓ Analysis complete!"
echo ""

# Print summary
echo "================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================"
echo ""
echo "Results locations:"
echo "  Training:   $EXPERIMENTS_DIR"
echo "  Evaluation: $EVAL_DIR"
echo "  Analysis:   $ANALYSIS_DIR"
echo ""
echo "Key files to review:"
echo "  - $ANALYSIS_DIR/hyperparameter_analysis.csv"
echo "  - $EVAL_DIR/agent_comparison.csv"
echo "  - $ANALYSIS_DIR/training_curves_all.png"
echo ""
echo "Next steps:"
echo "  1. Review top performers in agent_comparison.csv"
echo "  2. Check training curves for convergence"
echo "  3. Refine hyperparameter ranges based on results"
echo "  4. Run another sweep with refined ranges"
echo ""

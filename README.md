# Golf Card Game

A Python implementation of the card game Golf, including a simulation/AI training component.

## Overview

Golf is a card game where players try to achieve the lowest score possible by strategically swapping and replacing cards in their hand. This implementation includes both the game logic and an AI training system using Q-learning.

## Features

- Complete Golf card game implementation
- AI training simulation using Q-learning
- Command-line interface
- Multiple AI agents can play against each other
- Configurable game parameters

## Requirements

- Python 3.x

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/golf.git
cd golf
```

## Usage

### Running the Simulation

To start the AI training simulation: 

```bash
python simulation.py
```

### Collecting Tensor Transitions for Offline RL

Enable the optional tensor logger to persist state/action/reward transitions that
can be replayed when training offline agents:

```bash
python -m src.simulation --games 100 --holes 9 --log-tensors --tensor-log-dir data
```

Each worker writes four files per prefix (`.npz`, `.json`, `_metrics.json`,
`_metrics_series.json`). The compressed archive stores:

- `states`: one-hot tensors with shape `(ranks, positions, suits)`
- `next_states`: successor tensors with identical shape
- `actions`: canonical action ids (0: take face card, 1: draw deck, 2-15: place/discard variants)
- `rewards`, `dones`: scalar outcomes per transition

Any run that enables logging prints a short diagnostics block highlighting
skewed action distributions or flat reward signals so you can catch degenerate
datasets early.

### Loading Tensor Datasets

Use `TensorTransitionDataset` to load and prepare artifacts for learning:

```python
from src.tensor_dataset import TensorTransitionDataset

dataset = TensorTransitionDataset("data/tensor_transitions")
batch = dataset.as_qtransformer_arrays()

# batch contains numpy arrays ready for torch/jax:
#   player_cards, holding_cards, discard_top,
#   next_player_cards, next_holding_cards, next_discard_top,
#   actions, rewards, dones
```

The helper automatically converts the 3D one-hot tensors to the token format
expected by `QTransformer` (six visible cards plus the current discard top,
with 52 representing an unknown card). Each `TensorTransitionRecord` also
exposes the original metadata and decoded `(action_num, action, position)`
tuple via `record.action_tuple()` when you need human-readable actions.

### Offline DQN Training

Train DQN agents on collected transitions:

```bash
# Single training run
python -m src.dqn_offline \
  --archive-prefix tmp/tensor_logs_batch/tensor_transitions_combined \
  --output-dir tmp/my_model \
  --epochs 20

# Hyperparameter sweep (train multiple agents)
python -m src.experiment_runner \
  --config configs/experiment_quick.json \
  --output-dir tmp/experiments \
  --epochs 20
```

### Evaluating DQN Agents

Test trained agents against baseline players:

```bash
# Evaluate all experiments
python -m src.evaluate_agents \
  --experiments-dir tmp/experiments \
  --output-dir tmp/evaluations \
  --games 100

# Evaluate single checkpoint
python -m src.evaluate_agents \
  --checkpoint tmp/experiments/exp_001_.../offline_dqn.pt \
  --games 100
```

This generates:
- `evaluation_report.txt` - Ranked models with performance, training metrics, and hyperparameters
- `agent_comparison.csv` - Tabular comparison of all agents
- Individual game results and summaries per experiment

### Analyzing Results

Generate training curves and hyperparameter analysis:

```bash
python -m src.analyze_experiments \
  --experiments-dir tmp/experiments \
  --output-dir tmp/analysis
```

This generates:
- `training_curves_all.png` - Loss curves for all experiments
- `hyperparameter_analysis.csv` - Performance vs hyperparameters
- Heatmaps showing parameter interactions

See `configs/README_TRAINING_PIPELINE.md` for detailed documentation.

## Game Parameters

You can modify various game parameters in the simulation:
- Number of players
- Number of rounds ("holes")
- Learning rate
- Exploration rate (epsilon)

## Game Rules

### Setup
- Each player is dealt 6 cards face down in a 2x3 grid
- One card is placed face-up to start the discard pile
- Remaining cards form the draw pile

### Gameplay
1. On their turn, a player can:
   - Draw from the deck
   - Take the top card from the discard pile
2. After drawing, they must either:
   - Replace one of their cards with the drawn card
   - Discard the drawn card and flip one their cards
3. The round ends when any of the players flipped of their cards
4. At the end of 9 rounds, the player with the lowest score wins

### Scoring
- Number cards (3-10): Face value
- Face cards (J, Q): 10 points
- 2: -2 points
- King: 0 points
- Ace: 1 point
- Matching cards in the same column: 0 points

## Project Structure

```
golf/
├── simulation.py      # Q-learning simulation
├── game.py           # Core game logic
├── player.py         # Player class definitions
└── README.md
```

## AI Implementation

The project uses Q-learning, a reinforcement learning algorithm, to train AI agents. The agents learn optimal card selection and replacement strategies through repeated gameplay.

### Key Components
- State representation: Current hand configuration
- Actions: Card selections and replacements
- Rewards: Based on final score and game outcome

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

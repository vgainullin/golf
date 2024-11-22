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

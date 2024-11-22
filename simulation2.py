from copy import deepcopy
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import json
from collections.abc import MutableSequence
from collections import namedtuple

# Define Card as a NamedTuple
Card = namedtuple('Card', ['rank', 'suit'])

# Define the Golf Deck
class GolfDeck(MutableSequence):
    def __init__(self, deck_type="French"):
        if deck_type == "French":
            ranks = [str(n) for n in range(2, 11)] + list('JQKA')
            suits = 'spades diamonds clubs hearts'.split()
            self._cards = [Card(rank, suit) for suit in suits for rank in ranks]
        elif deck_type == "Blank":
            self._cards = []
        else:
            self._cards = deck_type

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

    def __setitem__(self, position, value):
        self._cards[position] = value

    def __delitem__(self, position):
        del self._cards[position]

    def insert(self, position, value):
        self._cards.insert(position, value)

# Player Class
@dataclass
class Player:
    name: str
    id: int
    agent_type: str = 'Heuristic'
    score: int = 10
    cards: list = None
    open_cards: list = None
    holding: str = None
    last_action: str = None
    action_num: int = 0
    card_to_score: dict = None
    open_ranks: list = None
    game_state: list = None

    def __post_init__(self):
        self.cards = [["X", "X", "X"], ["X", "X", "X"]]
        self.open_cards = [["X", "X", "X"], ["X", "X", "X"]]
        self.card_to_score = dict(zip(
            [str(n) for n in range(3, 11)] + list("JQKA"), 
            list(range(3, 11)) + [10, 10, 0, 1]
        ))
        self.card_to_score["2"] = -2
        self.card_to_score["X"] = np.nan

    def calculate_score(self, final=False):
        scores = deepcopy(self.open_cards)
        for i in range(3):
            if self.open_cards[0][i] == self.open_cards[1][i] != "X":
                scores[0][i] = 0
                scores[1][i] = 0
            else:
                scores[0][i] = self.card_to_score[self.open_cards[0][i]]
                scores[1][i] = self.card_to_score[self.open_cards[1][i]]
        self.score = np.nansum(scores)

    def gather_game_state(self, face_card):
        self.game_state = list(np.array(self.open_cards).flatten())
        self.game_state.append(face_card[0])
        if self.holding:
            self.game_state.append(self.holding[0])
        else:
            self.game_state.append('0')
        self.game_state = ''.join(self.game_state)

# RL Agent Class
class RLGolfAgent:
    def __init__(self, player_id, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.player_id = player_id
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        return max(self.q_table.get(state, {}), key=lambda x: self.q_table[state].get(x, 0), default=random.choice(available_actions))

    def update_q_value(self, state, action, reward, next_state):
        old_q_value = self.q_table.get(state, {}).get(action, 0)
        next_max_q_value = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q_value - old_q_value)
        self.q_table.setdefault(state, {})[action] = new_q_value

# Golf Game Environment
class Golf:
    def __init__(self, players):
        self.deck = GolfDeck("French")
        self.discard = GolfDeck("Blank")
        self.face_card = None
        self.players = players
        self.last_turn = False
        self.game_over = False
        self.end_game_player_id = None

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        for player in self.players:
            for row in range(2):
                for col in range(3):
                    player.cards[row][col] = self.deck.pop()
        self.face_card = self.deck.pop()
        self.discard.append(self.face_card)

    def take_action(self, player, action, position=None):
        if action == "take_new":
            player.holding = self.deck.pop()
        elif action == "take_face":
            player.holding = self.discard.pop()
        elif action == "place" and position:
            discard_card = player.cards[position[0]][position[1]]
            self.discard.append(discard_card)
            player.cards[position[0]][position[1]] = player.holding
            player.holding = None
        elif action == "discard":
            self.discard.append(player.holding)
            player.holding = None
        player.calculate_score()
        return player.score

# Training and Testing Framework
class GolfRLTraining:
    def __init__(self, num_games=1000, max_rounds=100, rank_cutoff=5):
        self.num_games = num_games
        self.max_rounds = max_rounds
        self.rank_cutoff = rank_cutoff
        self.q_table = {}

    def run_simulation(self, agent_types):
        ledger = []
        for game_num in range(self.num_games):
            players = [Player(name=f"PL{i+1}", id=i, agent_type=agent_types[i]) for i in range(len(agent_types))]
            agents = {p.id: RLGolfAgent(p.id) for p in players if p.agent_type == 'RL'}
            golf = Golf(players=players)
            golf.shuffle()
            golf.deal()

            for round_num in range(self.max_rounds):
                if golf.last_turn:
                    break

                for player in players:
                    if player.agent_type == "RL":
                        agent = agents[player.id]
                        player.gather_game_state(golf.face_card)
                        state = player.game_state
                        available_actions = ["take_new", "take_face", "place", "discard"]
                        action = agent.choose_action(state, available_actions)
                        reward = golf.take_action(player, action)
                        player.gather_game_state(golf.face_card)
                        next_state = player.game_state
                        agent.update_q_value(state, action, reward, next_state)
                    elif player.agent_type == "Heuristic":
                        # Define Heuristic behavior if needed
                        pass
                    elif player.agent_type == "Random":
                        action = random.choice(["take_new", "take_face", "place", "discard"])
                        golf.take_action(player, action)

            for player in golf.players:
                player.calculate_score(final=True)
                ledger.append(dict(player_id=player.id, score=player.score, game=game_num))

        self.save_ledger(ledger)

    def save_ledger(self, ledger, filename='game_results.csv'):
        df = pd.DataFrame(ledger)
        df.to_csv(filename, index=False)

    def save_q_table(self, filename='q_table.json'):
        with open(filename, 'w') as file:
            json.dump(self.q_table, file)

# Main function
def main():
    training = GolfRLTraining()
    training.run_simulation(['RL', 'Heuristic', 'Random', 'RL'])
    training.save_q_table()

if __name__ == "__main__":
    main()

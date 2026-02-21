from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from collections import namedtuple
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import json
import multiprocessing as mp
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for certain test environments
    torch = None

import collections
from collections.abc import MutableSequence
from collections import namedtuple
from collections import deque
from src.qtransformer import QTransformer, ReplayBuffer, CardEmbedding, train_episode
from src.tensor_logger import TensorTransitionLogger, decode_action_id
from src.tensor_dataset import tensor_to_player_tokens

if TYPE_CHECKING:  # pragma: no cover
    from .offline_agent import OfflineDQAgent


Card = namedtuple('Card', ['rank', 'suit'])

DEFAULT_NUM_GAMES = 1
DEFAULT_HOLES_PER_GAME = 1
DEFAULT_TENSOR_LOG_PREFIX = "tensor_transitions"

rank_cutoff = 4
verbose = True

@dataclass(frozen=True)
class SimulationConfig:
    num_games: int = DEFAULT_NUM_GAMES
    holes_per_game: int = DEFAULT_HOLES_PER_GAME
    shuffle: bool = True
    verbose: bool = False
    rank_cutoff: int = 4
    output_dir: Optional[str] = None
    log_tensors: bool = False
    tensor_log_dir: Path | str = Path("data")
    tensor_log_prefix: str = DEFAULT_TENSOR_LOG_PREFIX
    dqn_player_id: Optional[int] = None
    dqn_checkpoint: Optional[str] = None
    dqn_device: str = "auto"


@dataclass
class SimulationResult:
    worker_id: int
    seed: int
    ledger: List[Dict[str, Any]]
    q_table: Dict[str, Dict[str, float]]
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: List[str] = field(default_factory=list)
    shuffle_history: List[Tuple[Tuple[str, str], ...]] = field(default_factory=list)


@dataclass
class AggregationResult:
    ledger: List[Dict[str, Any]]
    avg_scores: Dict[int, float]
    artifact_paths: List[str]
    worker_count: int
    worker_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)

class GolfDeck(MutableSequence):
    
    def __init__(self, cards="French"):
        self.ranks = [str(n) for n in range(2, 10)] + list('XJQKA')
        self.suits = 'spades diamonds clubs hearts'.split()
        
        if cards == "French":
            self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        elif cards == "2xFrench":
            self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks] * 2
        elif cards == "Blank":
            self._cards = []
        else:
            raise ValueError(f"Invalid deck type: {cards}")

    def card2index(self, card, ignore_suit=False):
        """Convert a card to its unique original index in the unshuffled deck.
        
        If ignore_suit is True, return the index based only on the rank.
        If the card is unknown, return an index outside the normal range.
        """
        try:
            rank_index = self.ranks.index(card.rank)
            if ignore_suit:
                return rank_index
            else:
                suit_index = self.suits.index(card.suit)
                return suit_index * len(self.ranks) + rank_index
        except (AttributeError, ValueError):
            # Return an index outside the normal range for unknown cards
            return len(self.ranks) * len(self.suits) + 1

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
    

class Player:
    def __init__(self, name, id, type='Heuristic'):
        self.name = name
        self.id = id
        self.type = type
        self.score= 10 # TODO: Initialize with maximum value? 10
        self.reward = 0
        self.cards = [
            ["?", "?", "?"],
            ["?", "?", "?"]
            ]
        self.open_cards = [
            ["?", "?", "?"],
            ["?", "?", "?"]
            ]
        self.scores = [
            ["?", "?", "?"],
            ["?", "?", "?"]
            ]
        self.open_ranks = [
            ["?", "?", "?"],
            ["?", "?", "?"]
            ]
        self.holding = None
        self.last_action = None
        self.action_num = 0
        card_to_score = dict(zip([str(n) for n in range(3, 10)] + list("XJQKA"), list(range(3, 10))+[10, 10, 10, 0, 1]))
        card_to_score["2"] = -2
        card_to_score["?"] = np.nan
        self.card_to_score = card_to_score
        self.calculate_score()
        self.open_ranks = [
            ["?", "?", "?"],
            ["?", "?", "?"]
            ]
        self.game_state = []

        if self.type == 'CountingHeuristic':
            self.card_counts = None
            self.total_cards = 52
            self.probability_threshold = 0.4  # Default value, can be changed

    def initialize_card_counts(self):
        self.card_counts = collections.Counter()
        ranks = [str(n) for n in range(2, 10)] + list('XJQKA')
        for rank in ranks:
            self.card_counts[rank] = 4

    def update_card_counts(self, cards, face_card):
        for card in cards:
            if card != '?':
                rank = card[0]
                self.card_counts[rank] -= 1
                self.total_cards -= 1
        if face_card != '?':
            rank = face_card[0]
            self.card_counts[rank] -= 1
            self.total_cards -= 1

    def get_card_probabilities(self):
        probabilities = {}
        for rank, count in self.card_counts.items():
            probabilities[rank] = count / self.total_cards
        return probabilities

    def card2rank(self, card):
        if card is None or card == "?":
            return card
        else:
            return card[0]
            
    def get_card_ranks(self, cards):
        open_ranks = [
            [self.card2rank(j) for j in i] for i in cards
        ]
        return open_ranks

    def score_cards(self, cards):
        scores = deepcopy(cards)
        for i in range(3):
            if cards[0][i] == cards[1][i] != "?":
                scores[0][i] = 0
                scores[1][i] = 0
            else:
                if cards[0][i] is not None:
                    scores[0][i] = self.card_to_score.get(cards[0][i], np.nan)
                else:
                    scores[0][i] = np.nan

                if cards[1][i] is not None:
                    scores[1][i] = self.card_to_score.get(cards[1][i], np.nan)
                else:
                    scores[1][i] = np.nan

        scores = np.array(scores)
        score = np.nansum(scores)
        return score, scores

    def calculate_score(self, final=False):
        if final:
            self.open_ranks = self.get_card_ranks(self.cards)
        else:
            self.open_ranks = self.get_card_ranks(self.open_cards)
        self.score, self.scores = self.score_cards(self.open_ranks)
    
    def gather_game_state(self, game=None):
        self.game_state = list(np.array(self.open_ranks).flatten())
        self.game_state_tokens = self.open_cards
        # flatten a nested list
        self.game_state_tokens = [item for sublist in self.game_state_tokens for item in sublist]
        if game:
            self.game_state_tokens = [game.deck.card2index(card) for card in self.game_state_tokens]
            self.game_state_tokens.append(game.deck.card2index(game.face_card))
            self.game_state.append(self.card2rank(game.face_card))
            # add next player id
            #self.game_state += list(np.array(game.players[(self.id + 1) % game.num_players].open_ranks).flatten())
            
        if self.holding:
            self.game_state.append(self.card2rank(self.holding))
        else:
            self.game_state.append('0')
        self.game_state = ''.join(self.game_state)


class Golf:
    @staticmethod
    def pos_index(player_idx, row, col):
        """Map (player, row, col) to flat position index."""
        return player_idx * 7 + (row * 3 + col)

    def encode_golf_tensor(self):
        """
        Returns a tensor (numpy array) of shape (num_ranks, num_positions, num_suits), where:
        - num_ranks: number of unique ranks (13 for standard deck)
        - num_positions: 7 per player (6 slots + 1 holding) + 3 (deck, discard, face)
        - num_suits: number of suits (4 for standard deck)
        Each cell [rank, pos, suit] is 1 if that card is at that position, else 0.
        Helper functions are used for clarity.
        """
        num_ranks = len(self.deck.ranks)
        num_suits = len(self.deck.suits)
        num_players = self.num_players
        num_positions = num_players * 7 + 3
        tensor = np.zeros((num_ranks, num_positions, num_suits), dtype=np.int8)

        def get_indices(card):
            if card == "?" or card is None:
                return None, None
            try:
                rank_idx = self.deck.ranks.index(card.rank)
                suit_idx = self.deck.suits.index(card.suit)
                return rank_idx, suit_idx
            except Exception:
                return None, None


        # Player open slots
        for p, player in enumerate(self.players):
            for row in range(2):
                for col in range(3):
                    card = player.cards[row][col]
                    rank_idx, suit_idx = get_indices(card)
                    if rank_idx is not None and suit_idx is not None:
                        col_idx = Golf.pos_index(p, row, col)
                        tensor[rank_idx, col_idx, suit_idx] = 1
            # Player holding
            if player.holding and player.holding != "?":
                rank_idx, suit_idx = get_indices(player.holding)
                if rank_idx is not None and suit_idx is not None:
                    col_idx = p * 7 + 6
                    tensor[rank_idx, col_idx, suit_idx] = 1

        # Deck
        for card in self.deck:
            rank_idx, suit_idx = get_indices(card)
            if rank_idx is not None and suit_idx is not None:
                tensor[rank_idx, num_players * 7, suit_idx] = 1

        # Discard pile (all except face card)
        if len(self.discard) > 0:
            for card in self.discard:
                if card == self.face_card:
                    continue
                rank_idx, suit_idx = get_indices(card)
                if rank_idx is not None and suit_idx is not None:
                    tensor[rank_idx, num_players * 7 + 1, suit_idx] = 1

        # Face card (top of discard)
        if self.face_card:
            rank_idx, suit_idx = get_indices(self.face_card)
            if rank_idx is not None and suit_idx is not None:
                tensor[rank_idx, num_players * 7 + 2, suit_idx] = 1

        return tensor
    def __init__(self, players=None, deck_type="French", verbose=False):
        self.deck = GolfDeck(cards=deck_type)
        self.discard = GolfDeck(cards="Blank")
        self.face_card = None
        self.players = players
        self.num_players = len(players)
        self.last_turn = False
        self.game_over = False
        self.end_game_player_id = None
        self.verbose = verbose
        self.last_shuffle_signature: Tuple[Tuple[str, str], ...] = tuple()
        
    def shuffle(self):
        random.shuffle(self.deck)
        self.last_shuffle_signature = tuple((card.rank, card.suit) for card in list(self.deck)[:5])

    def deal(self):
        for player in self.players:
            for row in range(2):
                for col in range(3):
                    player.cards[row][col] = self.deck.pop()
        face_card = self.deck.pop()
        self.discard.append(face_card)
        self.face_card = face_card
        
    
    def take_action(self, player_id, action_array):
        # step 1: revealed_card = choice (deck.pop(), discard.pop())
        # step 2: if revealed_card != face_card:
        #            -> choice(position, discard), else choice(position)
        # action = [turn_num, action, position_int]
        score_before_action = self.players[player_id].score
        available_actions = {
        0: ['take_face_card', 'take_new'],
        1: ['place','discard']
        }
        action_num = action_array[0]
        action_term = available_actions[action_array[0]][action_array[1]]
        if action_array[2] == None:
            pos_tuple = None
        else:
            pos_tuple = action_array[2] // 3, action_array[2] % 3
        #print(f"{action_array[2]} -> {pos_tuple}")
        
        if self.last_turn and self.end_game_player_id == player_id:
            self.game_over = True
            print("Game Over")
            return 0  # Return a default reward
        self.players[player_id].last_action = action_array
        if action_num == 0:
            if action_term == "take_new":
                self.players[player_id].holding = self.deck.pop()
                self.players[player_id].action_num = 1

            elif action_term == "take_face_card":
                if len(self.discard) > 0:  # Check if discard pile is not empty
                    self.players[player_id].holding = self.discard.pop()
                    self.players[player_id].action_num = 1
                else:
                    print(f"ERROR: Discard pile is empty, cannot take face card {player_id}")
                    return 0  # Return a default reward
            else:
                print("Incorrect action")
                
        elif action_num == 1:
            if self.game_over:
                print("ERROR Game ended")
                return 0  # Return a default reward
            elif not self.players[player_id].holding:
                print(f"ERROR, Player should be holding a card {player_id}")
                return 0  # Return a default reward
            elif action_term == "place" and pos_tuple != None:
                # remove the card that is being replaced
                discard_ = self.players[player_id].cards[pos_tuple[0]][pos_tuple[1]]
                # place it in discard pile
                self.discard.append(discard_)
                # put the new card into its new position
                self.players[player_id].cards[pos_tuple[0]][pos_tuple[1]] = self.players[player_id].holding
                self.players[player_id].open_cards[pos_tuple[0]][pos_tuple[1]] = self.players[player_id].holding
                self.players[player_id].holding = None
                self.players[player_id].action_num = 0
                self.face_card = discard_

            elif action_term == "discard" and pos_tuple:
                # Do not place a card, instead flip a new card
                if self.players[player_id].open_cards[pos_tuple[0]][pos_tuple[1]] != "?":
                    print("ERROR Already flipped this card")
                else:
                    self.discard.append(self.players[player_id].holding)
                    self.face_card = self.players[player_id].holding
                    self.players[player_id].holding = None
                    self.players[player_id].open_cards[pos_tuple[0]][pos_tuple[1]] = self.players[player_id].cards[pos_tuple[0]][pos_tuple[1]]
                    self.players[player_id].action_num = 0
            else:
                print("ERROR No condition met")
                return 0  # Return a default reward
        self.players[player_id].calculate_score()
        reward = score_before_action - self.players[player_id].score
        return reward

def encode_pos_tuple(pos):
    if pos:
        # (0, 0): 0
        # (0, 1): 1
        # (0, 2): 2
        # (1, 0): 3
        # (1, 1): 4
        # (1, 2): 5
        if pos[0] == 1:
            pos_int = (np.array(pos) + 1).sum()
        else:
            pos_int = np.array(pos).sum()
    else:
        pos_int = None
    return pos_int


def calc_opt_heuristic_position(player, holding_card, random_action=False):
    """
    Calculates the optimal position to place the holding card for a given player.
    Returns the optimal position (row, col) and the updated score if the card is placed there.
    If random_action is True, returns a random available position and the corresponding updated score.
    """
    min_score = 99
    opt_pos = None
    upd_score = None
    available_pos = []

    for row in range(2):
        for col in range(3):
            if player.open_cards[row][col] == "?":
                available_pos.append((row, col))
                player_cards_copy = deepcopy(player.open_cards)
                player_cards_copy[row][col] = holding_card
                #print("player_cards_copy:", player_cards_copy)
                player_card_ranks = player.get_card_ranks(player_cards_copy)
                score, _ = player.score_cards(player_card_ranks)

                if score < min_score:
                    min_score = score
                    opt_pos = (row, col)
                    upd_score = score

    if random_action and available_pos:
        rand_pos = random.choice(available_pos)
        player_cards_copy = deepcopy(player.open_cards)
        player_cards_copy[rand_pos[0]][rand_pos[1]] = holding_card
        player_card_ranks = player.get_card_ranks(player_cards_copy)
        upd_score, _ = player.score_cards(player_card_ranks)
        opt_pos = rand_pos

    return opt_pos, upd_score

def get_player_action(game, player_id, action_num, rank_cutoff=5, take_random_action=False):
    available_actions = {
        0: ['take_face_card', 'take_new'],
        1: ['place', 'discard']
    }
    player = game.players[player_id]

    # Action 0: Take a card
    if action_num == 0:
        if player.type == 'CountingHeuristic':
            card_probabilities = player.get_card_probabilities()
            rank_of_face_card = player.card2rank(game.face_card)
            face_card_probability = card_probabilities[rank_of_face_card]
            lower_rank_probability = sum(card_probabilities[rank] for rank in card_probabilities if player.card_to_score[rank] < player.card_to_score[rank_of_face_card])
            if lower_rank_probability > player.probability_threshold:
                return 1, None, 0  # Take a new card from the deck
            else:
                return 0, None, 0  # Take the face card
        elif player.type == 'Heuristic':
            rank_of_face_card = player.card2rank(game.face_card)
            rank_match = np.argwhere(player.open_ranks == rank_of_face_card)
            if player.card_to_score[game.face_card[0]] < rank_cutoff or rank_match.size > 0:
                return 0, None, 0  # Take the face card
            else:
                return 1, None, 0  # Take a new card from the deck
        elif take_random_action:
            return random.choice([0, 1]), None, 0
        else:
            raise TypeError(f"Invalid player type: {player.type}")

    # Action 1: Place or discard a card
    elif action_num == 1:
        current_score = player.score
        opt_pos, upd_score = calc_opt_heuristic_position(player, player.holding)
        reward = current_score - upd_score

        if take_random_action:
            rand_action = random.choice([0, 1])
            if rand_action == 0:
                rand_pos, upd_rand_score = calc_opt_heuristic_position(player, player.holding, random_action=True)
                reward = current_score - upd_rand_score
                return rand_action, encode_pos_tuple(rand_pos), reward
            else:
                available_pos_to_place = np.argwhere(np.isnan(player.scores))
                if len(available_pos_to_place) > 0:
                    rand_pos = tuple(available_pos_to_place[random.randint(0, len(available_pos_to_place) - 1)])
                else:
                    rand_pos = None
                return rand_action, encode_pos_tuple(rand_pos), 0
        else:
            available_pos_to_place = np.argwhere(np.isnan(player.scores))
            can_place = len(available_pos_to_place) > 0

            if upd_score <= (current_score - rank_cutoff):
                action, pos = 0, opt_pos
            elif can_place and (upd_score - current_score) < rank_cutoff:
                action, pos = 0, tuple(available_pos_to_place[0])
            elif can_place:
                action, pos = 1, tuple(available_pos_to_place[0])
            else:
                raise ValueError("No valid action found")

            return action, encode_pos_tuple(pos), reward

    else:
        raise ValueError(f"Invalid action number: {action_num}")

# Define Q-learning parameters
epsilon = 0.1  # Exploration rate
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

def play_single_turn(
    golf,
    player_id,
    action_num,
    Q,
    state_,
    rank_cutoff,
    *,
    game_num,
    hole,
    round_num,
    transition_logger: TensorTransitionLogger | None = None,
    offline_agent: "OfflineDQAgent | List[OfflineDQAgent | None] | None" = None,
):
    player = golf.players[player_id]
    state_tensor = None
    if hasattr(golf, "encode_golf_tensor") and (
        transition_logger is not None or player.type == 'OfflineDQN'
    ):
        state_tensor = golf.encode_golf_tensor()

    # Determine action based on player type
    if player.type == 'Heuristic':
        action, pos, reward = get_player_action(deepcopy(golf), player_id, action_num, rank_cutoff, take_random_action=False)
    elif player.type == 'Random':
        action, pos, reward = get_player_action(deepcopy(golf), player_id, action_num, rank_cutoff, take_random_action=True)
    elif player.type == 'CountingHeuristic':
        action, pos, reward = get_player_action(deepcopy(golf), player_id, action_num, rank_cutoff, take_random_action=False)
    elif player.type == 'RL':
        q_state_ = Q.get(f'{action_num}'+state_, {}).items()
        if q_state_:
            action_pos = max(q_state_, key=lambda x: x[1])[0]
            action, pos = action_pos.split("|")
            action = int(action)
            pos = None if pos == 'None' else int(pos)
        else:
            action, pos, reward = get_player_action(deepcopy(golf), player_id, action_num, rank_cutoff, take_random_action=True)
    elif player.type == 'OfflineDQN':
        # Support both single agent and array of agents
        if isinstance(offline_agent, list):
            agent = offline_agent[player_id]
        else:
            agent = offline_agent

        if agent is None:
            raise RuntimeError("Offline DQN agent not provided. Pass offline_agent parameter to play_single_turn.")
        if state_tensor is None:
            if not hasattr(golf, "encode_golf_tensor"):
                raise RuntimeError("Golf environment does not support tensor encoding required by Offline DQN agent.")
            state_tensor = golf.encode_golf_tensor()
        cards, holding, discard_top = tensor_to_player_tokens(
            state_tensor,
            player_id=player_id,
            num_players=golf.num_players,
        )
        state_tokens = np.concatenate(
            (
                cards.astype(np.int64, copy=False),
                np.asarray([holding, discard_top], dtype=np.int64),
            )
        )
        action_id = agent.select_action(state_tokens, action_num)

        # Validate action_id range (0-15 for NUM_ACTIONS=16)
        if not (0 <= action_id < 16):
            action, pos, reward = get_player_action(
                deepcopy(golf),
                player_id,
                action_num,
                rank_cutoff,
                take_random_action=True,
            )
        else:
            sel_action_num, action, position = decode_action_id(action_id)
            if sel_action_num != action_num:
                raise RuntimeError(
                    f"Offline DQN agent produced action for stage {sel_action_num} during stage {action_num}."
                )
            pos = None if position is None else int(position)
            valid = True
            if action_num == 0:
                if action == 0 and len(golf.discard) == 0:
                    valid = False
                elif action == 1 and len(golf.deck) == 0:
                    valid = False
            elif action_num == 1:
                if action == 0 and pos is None:
                    valid = False
                elif action == 1:
                    if pos is None:
                        valid = False
                    else:
                        row, col = divmod(pos, 3)
                        if player.open_cards[row][col] != "?":
                            valid = False
            if not valid:
                action, pos, reward = get_player_action(
                    deepcopy(golf),
                    player_id,
                    action_num,
                    rank_cutoff,
                    take_random_action=True,
                )
            else:
                reward = 0
    else:
        raise ValueError(f"Error: Player type '{player.type}' not recognized.")

    # Execute action
    action_array = [action_num, action, pos]
    reward = golf.take_action(player_id=player_id, action_array=action_array)

    # Get new state
    golf.players[player_id].gather_game_state(golf)
    new_state = golf.players[player_id].game_state

    if transition_logger is not None and state_tensor is not None and hasattr(golf, "encode_golf_tensor"):
        next_state_tensor = golf.encode_golf_tensor()
        transition_logger.log(
            state=state_tensor,
            next_state=next_state_tensor,
            reward=reward,
            done=golf.game_over,
            metadata={
                "game": game_num,
                "hole": hole,
                "round": round_num,
                "player_id": player_id,
                "action_num": action_num,
                "action": action,
                "position": pos,
            },
        )

    # Update Q-table for RL player
    if player.type == 'RL':
        update_q_table(Q, action_num, state_, new_state, action, pos, reward)
        player.reward += reward
    else:
        player.reward += reward
    return new_state, reward, action_num

def update_q_table(Q, action_num, state, new_state, action, pos, reward):
    old_q_value = Q.get(f'{action_num}'+state, {}).get(f"{action}|{pos}", 0)
    next_max_q_value = max(Q.get(f'{action_num}'+new_state, {}).values(), default=0)
    new_q_value = old_q_value + alpha * (reward + gamma * next_max_q_value - old_q_value)
    Q.setdefault(f'{action_num}'+state, {})[f"{action}|{pos}"] = new_q_value

def play_game(
    golf,
    game_num,
    hole,
    Q,
    model,
    rank_cutoff: int = 4,
    verbose: bool = False,
    shuffle: bool = True,
    transition_logger: TensorTransitionLogger | None = None,
    offline_agent: "OfflineDQAgent | List[OfflineDQAgent | None] | None" = None,
):
    initial_shuffle_signature: Tuple[Tuple[str, str], ...] = tuple()
    if shuffle:
        golf.shuffle()
        initial_shuffle_signature = golf.last_shuffle_signature
    golf.deal()

    # Initialize CountingHeuristic players
    for player in golf.players:
        if player.type == 'CountingHeuristic':
            player.initialize_card_counts()
            for other_player in golf.players:
                player.update_card_counts(other_player.cards[0] + other_player.cards[1], golf.face_card)

    # Play rounds
    round_num = 0
    while not golf.game_over:
        if torch is not None and model is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            replay_buffer = ReplayBuffer(capacity=10000)
        else:
            optimizer = None
            replay_buffer = None

        epsilon = 0.5
        batch_size = 32
        episode_rewards = []

        # iterate over all players
        for player_id in range(golf.num_players):
            golf.players[player_id].gather_game_state(golf)
            state_ = golf.players[player_id].game_state

            if '?' not in golf.players[player_id].game_state:
                golf.game_over = True
                break

            upd_state_0, reward_0, action_0 = play_single_turn(
                golf,
                player_id,
                0,
                Q,
                state_,
                rank_cutoff,
                game_num=game_num,
                hole=hole,
                round_num=round_num,
                transition_logger=transition_logger,
                offline_agent=offline_agent,
            )

            golf.players[player_id].gather_game_state(golf)

            upd_state_1, reward_1, action_1 = play_single_turn(
                golf,
                player_id,
                1,
                Q,
                upd_state_0,
                rank_cutoff,
                game_num=game_num,
                hole=hole,
                round_num=round_num,
                transition_logger=transition_logger,
                offline_agent=offline_agent,
            )

            if len(golf.deck) < golf.num_players + 2:
                if verbose:
                    print(f"Deck is empty, creating new deck {player_id}")
                golf.deck = GolfDeck()
                golf.shuffle()
                golf.deal()

            if verbose:
                print(
                    game_num,
                    hole,
                    round_num,
                    len(golf.deck),
                    player_id,
                    golf.players[player_id].score,
                    golf.players[player_id].game_state,
                )
            if '?' not in golf.players[player_id].game_state:
                golf.last_turn = True
                golf.end_game_player_id = player_id

        round_num += 1

    game_results: List[Dict[str, Any]] = []
    for player in golf.players:
        player.calculate_score(final=True)
        game_results.append(
            dict(
                player_id=player.id,
                score=player.score,
                hole=hole,
                game=game_num,
                reward=player.reward,
            )
        )

    return game_results, initial_shuffle_signature

def _init_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _default_player_roster(dqn_player_id: Optional[int] = None) -> List[Player]:
    roster = [
        Player(name="PL1", id=0, type='Heuristic'),
        Player(name="PL2", id=1, type='Heuristic'),
        Player(name="PL3", id=2, type='Heuristic'),
        Player(name="PL4", id=3, type='Heuristic'),
    ]
    if dqn_player_id is not None:
        if dqn_player_id < 0 or dqn_player_id >= len(roster):
            raise ValueError(f"dqn_player_id {dqn_player_id} out of range for roster size {len(roster)}")
        roster[dqn_player_id] = Player(
            name=f"DQN_{dqn_player_id}",
            id=dqn_player_id,
            type='OfflineDQN',
        )
    return roster


def run_simulation(
    config: SimulationConfig | None = None,
    *,
    num_games: Optional[int] = None,
    holes_per_game: Optional[int] = None,
    shuffle: Optional[bool] = None,
    log_tensors: Optional[bool] = None,
    tensor_log_dir: Path | str = Path("data"),
    tensor_log_prefix: str = DEFAULT_TENSOR_LOG_PREFIX,
    rank_cutoff_override: Optional[int] = None,
    verbose_override: Optional[bool] = None,
    seed: Optional[int] = None,
    worker_id: int = 0,
    game_offset: int = 0,
    output_dir: Optional[str] = None,
) -> SimulationResult | Dict[str, Any]:
    config_provided = config is not None
    if config is None:
        config = SimulationConfig(
            num_games=num_games if num_games is not None else DEFAULT_NUM_GAMES,
            holes_per_game=holes_per_game if holes_per_game is not None else DEFAULT_HOLES_PER_GAME,
            shuffle=True if shuffle is None else shuffle,
            verbose=verbose_override if verbose_override is not None else bool(verbose),
            rank_cutoff=rank_cutoff_override if rank_cutoff_override is not None else 4,
            output_dir=str(output_dir) if output_dir else None,
            log_tensors=bool(log_tensors),
            tensor_log_dir=tensor_log_dir,
            tensor_log_prefix=tensor_log_prefix,
            dqn_player_id=None,
            dqn_checkpoint=None,
            dqn_device="auto",
        )

    effective_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    _init_random_seed(effective_seed)

    effective_output = output_dir if output_dir is not None else config.output_dir

    ledger: List[Dict[str, Any]] = []
    q_table: Dict[str, Dict[str, float]] = {}
    shuffle_history: List[Tuple[Tuple[str, str], ...]] = []

    model = QTransformer() if torch is not None else None

    tensor_logger: TensorTransitionLogger | None = None
    tensor_log_dir_path: Path | None = None
    tensor_log_prefix_val: str | None = None
    if config.log_tensors:
        tensor_log_dir_path = Path(config.tensor_log_dir)
        tensor_log_dir_path.mkdir(parents=True, exist_ok=True)
        tensor_logger = TensorTransitionLogger(tensor_log_dir_path)
        if config_provided:
            tensor_log_prefix_val = f"{config.tensor_log_prefix}_worker_{worker_id}"
        else:
            tensor_log_prefix_val = config.tensor_log_prefix
    agent = None
    if config.dqn_checkpoint and config.dqn_player_id is not None:
        from src.offline_agent import OfflineDQAgent

        agent = OfflineDQAgent(
            Path(config.dqn_checkpoint),
            device=config.dqn_device,
        )

    assigned_dqn_player_id = config.dqn_player_id if agent is not None else None

    total_games = max(config.num_games, 0)
    for local_game in range(total_games):
        game_number = game_offset + local_game
        for hole in range(1, config.holes_per_game + 1):
            players = _default_player_roster(assigned_dqn_player_id)
            golf = Golf(players=list(players), deck_type="French", verbose=config.verbose)
            game_results, shuffle_signature = play_game(
                golf,
                game_number,
                hole,
                q_table,
                model,
                rank_cutoff=config.rank_cutoff,
                verbose=config.verbose,
                shuffle=config.shuffle,
                transition_logger=tensor_logger,
                offline_agent=agent,
            )
            ledger.extend(game_results)
            if shuffle_signature:
                shuffle_history.append(shuffle_signature)

    metrics = {
        "num_games": total_games,
        "holes_per_game": config.holes_per_game,
        "records": len(ledger),
    }
    if tensor_logger is not None:
        metrics["tensor_transitions"] = len(tensor_logger)

    artifact_paths: List[str] = []
    if effective_output:
        out_dir = Path(effective_output)
        out_dir.mkdir(parents=True, exist_ok=True)
        if ledger:
            ledger_path = out_dir / f"ledger_worker_{worker_id}.csv"
            pd.DataFrame(ledger).to_csv(ledger_path, index=False)
            artifact_paths.append(str(ledger_path))
        q_path = out_dir / f"q_table_worker_{worker_id}.json"
        with open(q_path, "w") as fp:
            json.dump(q_table, fp)
        artifact_paths.append(str(q_path))

    if tensor_logger is not None and tensor_log_dir_path is not None and tensor_log_prefix_val is not None:
        tensor_logger.save(prefix=tensor_log_prefix_val)
        base = tensor_log_dir_path / tensor_log_prefix_val
        artifact_paths.extend(
            [
                str(base.with_suffix(".npz")),
                str(base.with_suffix(".json")),
                str(base.with_name(f"{tensor_log_prefix_val}_metrics.json")),
                str(base.with_name(f"{tensor_log_prefix_val}_metrics_series.json")),
            ]
        )

    result = SimulationResult(
        worker_id=worker_id,
        seed=effective_seed,
        ledger=ledger,
        q_table=q_table,
        metrics=metrics,
        artifact_paths=artifact_paths,
        shuffle_history=shuffle_history,
    )

    if config_provided:
        return result

    legacy_payload = {
        "Q": q_table,
        "ledger": ledger,
        "all_game_results": [],
        "transition_logger": tensor_logger,
    }
    return legacy_payload


def _worker_entry(
    worker_id: int,
    config: SimulationConfig,
    seed: int,
    game_offset: int,
    output_dir: Optional[str],
    queue: mp.Queue,
) -> None:
    result = run_simulation(
        config=config,
        seed=seed,
        worker_id=worker_id,
        game_offset=game_offset,
        output_dir=output_dir,
    )
    queue.put(result)


def _resolve_worker_seed(base_seed: Optional[int], worker_id: int) -> int:
    if base_seed is None:
        return random.randint(0, 2**32 - 1)
    return base_seed + worker_id


def run_simulations_concurrently(
    config: SimulationConfig,
    num_workers: int = 1,
    base_seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> List[SimulationResult]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    effective_output = output_dir if output_dir is not None else config.output_dir

    if num_workers == 1:
        seed = _resolve_worker_seed(base_seed, 0)
        return [
            run_simulation(
                config=config,
                seed=seed,
                worker_id=0,
                game_offset=0,
                output_dir=effective_output,
            )
        ]

    total_games = max(config.num_games, 0)
    games_per_worker = [total_games // num_workers] * num_workers
    for idx in range(total_games % num_workers):
        games_per_worker[idx] += 1

    ctx = None
    queue = None
    if hasattr(mp, "get_context"):
        try:
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
        except (PermissionError, OSError):
            ctx = None
    if ctx is None:
        results: List[SimulationResult] = []
        game_offset = 0
        for worker_id, worker_games in enumerate(games_per_worker):
            seed = _resolve_worker_seed(base_seed, worker_id)
            worker_config = replace(config, num_games=worker_games)
            results.append(
                run_simulation(
                    config=worker_config,
                    seed=seed,
                    worker_id=worker_id,
                    game_offset=game_offset,
                    output_dir=effective_output,
                )
            )
            game_offset += worker_games
        results.sort(key=lambda item: item.worker_id)
        return results

    processes = []
    results: List[SimulationResult] = []
    game_offset = 0

    for worker_id, worker_games in enumerate(games_per_worker):
        seed = _resolve_worker_seed(base_seed, worker_id)
        worker_config = replace(config, num_games=worker_games)
        if worker_games == 0:
            results.append(
                run_simulation(
                    config=worker_config,
                    seed=seed,
                    worker_id=worker_id,
                    game_offset=game_offset,
                    output_dir=effective_output,
                )
            )
            continue

        process = ctx.Process(
            target=_worker_entry,
            args=(worker_id, worker_config, seed, game_offset, effective_output, queue),
        )
        processes.append(process)
        process.start()
        game_offset += worker_games

    for _ in processes:
        results.append(queue.get())

    for process in processes:
        process.join()

    results.sort(key=lambda item: item.worker_id)
    return results


def aggregate_worker_results(results: List[SimulationResult]) -> AggregationResult:
    combined_ledger: List[Dict[str, Any]] = []
    artifact_paths: List[str] = []
    worker_metrics: Dict[int, Dict[str, Any]] = {}
    for result in results:
        combined_ledger.extend(result.ledger)
        artifact_paths.extend(result.artifact_paths)
        worker_metrics[result.worker_id] = result.metrics

    if combined_ledger:
        df = pd.DataFrame(combined_ledger)
        avg_scores_series = df.groupby("player_id")["score"].mean()
        avg_scores = {int(k): float(v) for k, v in avg_scores_series.items()}
    else:
        avg_scores = {}

    return AggregationResult(
        ledger=combined_ledger,
        avg_scores=avg_scores,
        artifact_paths=artifact_paths,
        worker_count=len(results),
        worker_metrics=worker_metrics,
    )


def collect_tensor_artifacts(
    results: Iterable[SimulationResult],
    destination: Path,
    prefix: str = DEFAULT_TENSOR_LOG_PREFIX,
) -> Optional[TensorTransitionLogger]:
    bases: List[Path] = []
    seen: set[Path] = set()
    for result in results:
        for artifact in result.artifact_paths:
            path = Path(artifact)
            if path.suffix != '.npz':
                continue
            base = path.with_suffix('')
            if base in seen:
                continue
            metadata_path = base.with_suffix('.json')
            if not metadata_path.exists():
                continue
            seen.add(base)
            bases.append(base)

    if not bases:
        return None

    destination.mkdir(parents=True, exist_ok=True)
    combined_logger = TensorTransitionLogger(destination)

    for base in bases:
        npz_path = base.with_suffix('.npz')
        metadata_path = base.with_suffix('.json')
        if not npz_path.exists() or not metadata_path.exists():
            continue
        data = np.load(npz_path)
        states = data['states']
        next_states = data['next_states']
        rewards = data['rewards']
        dones = data['dones']
        metadata_entries = json.loads(metadata_path.read_text())

        for state, next_state, reward, done, meta in zip(
            states, next_states, rewards, dones, metadata_entries
        ):
            combined_logger.log(
                state=state,
                next_state=next_state,
                reward=float(reward),
                done=bool(done),
                metadata=meta,
            )

    combined_logger.save(prefix=prefix)
    return combined_logger


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Golf simulations sequentially or with multiprocessing workers and optional tensor logging.",
    )
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--games",
        "--num-games",
        dest="games",
        type=int,
        default=DEFAULT_NUM_GAMES,
        help="Number of games to simulate.",
    )
    parser.add_argument(
        "--holes",
        "--holes-per-game",
        dest="holes",
        type=int,
        default=DEFAULT_HOLES_PER_GAME,
        help="Number of holes per game.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes to spawn.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="Base seed for deterministic worker seeding.",
    )
    parser.add_argument(
        "--rank-cutoff",
        type=int,
        default=4,
        help="Rank cutoff heuristic for actions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to write aggregated artifacts.",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="Shuffle the deck before each game (default).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable deck shuffling between games.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose logging during simulations.",
    )
    parser.add_argument(
        "--log-tensors",
        action="store_true",
        help="Enable tensor transition logging.",
    )
    parser.add_argument(
        "--tensor-log-dir",
        type=Path,
        default=Path("data"),
        help="Directory where tensor transition artifacts will be written.",
    )
    parser.add_argument(
        "--tensor-log-prefix",
        default=DEFAULT_TENSOR_LOG_PREFIX,
        help="Filename prefix for tensor transition artifacts.",
    )
    parser.add_argument(
        "--dqn-player-id",
        type=int,
        help="Seat index (0-3) to control with the offline DQN agent.",
    )
    parser.add_argument(
        "--dqn-checkpoint",
        type=Path,
        help="Path to an offline DQN checkpoint to enable the agent.",
    )
    parser.add_argument(
        "--dqn-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device on which to run the offline DQN agent.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    global verbose
    verbose = args.verbose

    output_dir = args.output_dir
    dqn_checkpoint = args.dqn_checkpoint
    dqn_player_id = args.dqn_player_id
    if not dqn_checkpoint or dqn_player_id is None:
        dqn_checkpoint = None
        dqn_player_id = None
    config = SimulationConfig(
        num_games=max(args.games, 0),
        holes_per_game=max(args.holes, 0),
        shuffle=args.shuffle,
        verbose=bool(args.verbose),
        rank_cutoff=args.rank_cutoff,
        output_dir=str(output_dir) if output_dir else None,
        log_tensors=args.log_tensors,
        tensor_log_dir=args.tensor_log_dir,
        tensor_log_prefix=args.tensor_log_prefix,
        dqn_player_id=dqn_player_id,
        dqn_checkpoint=str(dqn_checkpoint) if dqn_checkpoint else None,
        dqn_device=args.dqn_device,
    )

    results = run_simulations_concurrently(
        config=config,
        num_workers=max(args.num_workers, 1),
        base_seed=args.base_seed,
        output_dir=str(output_dir) if output_dir else None,
    )
    aggregation = aggregate_worker_results(results)

    if output_dir:
        out_dir = output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if aggregation.ledger:
            combined_path = out_dir / "all_game_results.csv"
            pd.DataFrame(aggregation.ledger).to_csv(combined_path, index=False)
        consolidated_q_table = {f"worker_{res.worker_id}": res.q_table for res in results}
        q_path = out_dir / "q_table.json"
        with open(q_path, "w") as fp:
            json.dump(consolidated_q_table, fp)

    if aggregation.avg_scores:
        print("Average score per player:")
        for player_id in sorted(aggregation.avg_scores):
            score = aggregation.avg_scores[player_id]
            print(f"  Player {player_id}: {score:.2f}")
    else:
        print("No simulation data generated.")

    if config.log_tensors:
        tensor_path = Path(config.tensor_log_dir)
        for result in results:
            tensor_files = [Path(p) for p in result.artifact_paths if Path(p).suffix == '.npz']
            for tensor_file in tensor_files:
                print(f"Worker {result.worker_id} tensor log: {tensor_file}")
        combined_prefix = (
            config.tensor_log_prefix
            if args.num_workers <= 1
            else f"{config.tensor_log_prefix}_combined"
        )
        collected_logger = collect_tensor_artifacts(results, tensor_path, combined_prefix)
        if collected_logger is not None:
            metrics = collected_logger.metrics
            unique_count = metrics.get('unique_states', len(collected_logger))
            total_count = metrics.get('total_states', len(collected_logger))
            print(
                f"Collected {unique_count} unique transitions (total={total_count}) saved as '{combined_prefix}' in {tensor_path}"
            )
            diagnostics = collected_logger.diagnostics()
            if diagnostics.get("warnings"):
                print("Tensor log diagnostics:")
                for warning in diagnostics["warnings"]:
                    print(f"  - {warning}")
        else:
            print(f"No tensor artifacts found to collect in {tensor_path}")



if __name__ == "__main__":
    main()

from copy import copy, deepcopy
from dataclasses import dataclass
from collections import namedtuple
import random
import numpy as np
import pandas as pd
import json
import torch


import collections
from collections.abc import MutableSequence
from collections import namedtuple
from collections import deque
from src.qtransformer import QTransformer, ReplayBuffer, CardEmbedding, train_episode
from src.tensor_logger import TensorTransitionLogger

Card = namedtuple('Card', ['rank', 'suit'])

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
        
    def shuffle(self):
        random.shuffle(self.deck)

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
    *,
    game_num,
    hole,
    round_num,
    transition_logger: TensorTransitionLogger | None = None,
):
    player = golf.players[player_id]
    state_tensor = None
    if transition_logger is not None:
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
    else:
        raise ValueError(f"Error: Player type '{player.type}' not recognized.")

    # Execute action
    action_array = [action_num, action, pos]
    reward = golf.take_action(player_id=player_id, action_array=action_array)

    # Get new state
    golf.players[player_id].gather_game_state(golf)
    new_state = golf.players[player_id].game_state

    if transition_logger is not None and state_tensor is not None:
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

def play_game(golf, game_num, hole, Q, model, shuffle=True, transition_logger: TensorTransitionLogger | None = None):
    # Add this line
    if shuffle:
        golf.shuffle()
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        replay_buffer = ReplayBuffer(capacity=10000)
        
        epsilon = 0.5
        batch_size = 32
        episode_rewards = []

        # iterate over all players
        for player_id in range(golf.num_players):
            # Get initial state
            golf.players[player_id].gather_game_state(golf)
            state_ = golf.players[player_id].game_state
            init_state_tokens = np.array(golf.players[player_id].game_state_tokens)
            
            if '?' not in golf.players[player_id].game_state:
                golf.game_over = True
                break
            
            # Play first action (action_num = 0)
            upd_state_0, reward_0, action_0 = play_single_turn(
                golf,
                player_id,
                0,
                Q,
                state_,
                game_num=game_num,
                hole=hole,
                round_num=round_num,
                transition_logger=transition_logger,
            )
            
            golf.players[player_id].gather_game_state(golf)

            #replay_buffer.push(init_state_tokens, action_0, reward_0, np.array(golf.players[player_id].game_state_tokens), golf.game_over)
            # Play second action (action_num = 1)
            #if golf.players[player_id].type != 'RL':  # RL handled separately
            #loss = train_episode(model, optimizer, replay_buffer, batch_size)
            #print(loss)
            upd_state_1, reward_1, action_1 = play_single_turn(
                golf,
                player_id,
                1,
                Q,
                upd_state_0,
                game_num=game_num,
                hole=hole,
                round_num=round_num,
                transition_logger=transition_logger,
            )
            #check if there are cards left in the deck create a new deck otherwise use the discard pile
            if len(golf.deck) < golf.num_players + 2:
                if verbose:
                    print(f"Deck is empty, creating new deck {player_id}")
                golf.deck = GolfDeck()
                golf.shuffle() 
                golf.deal()

            # if np.random.random() < epsilon:
            #     action = np.random.randint(1)  # Random action
            # else:
            #     with torch.no_grad():
            #         state_tensor = torch.tensor(init_state_tokens[None, :], dtype=torch.long)
            #         q_values = model(state_tensor[:, :6], state_tensor[:, 6:])
            #         action = q_values.argmax().item()
            if verbose:
                print(game_num, hole, round_num, len(golf.deck), player_id,golf.players[player_id].score, golf.players[player_id].game_state)
            if '?' not in golf.players[player_id].game_state:
                golf.last_turn = True
                golf.end_game_player_id = player_id
            
        round_num += 1
    
    # Calculate final scores
    game_results = []
    for player in golf.players:
        player.calculate_score(final=True)
        game_results.append(dict(
            player_id=player.id,
            score=player.score,
            hole=hole,
            game=game_num,
            reward=player.reward
        ))
    
    return game_results

# Main execution
Q = {}
ledger = []
rank_cutoff = 4
verbose = True
num_games_to_simulate = 1
holes_per_game = 1
shuffle = True

all_game_results = []
model = QTransformer()
for game_num in range(num_games_to_simulate):
    for hole in range(1, holes_per_game + 1):
        # Initialize players as a deque for easy rotation
        players = deque([
            Player(name="PL1", id=0, type='Random'), # Random
            Player(name="PL2", id=1, type='Heuristic'), # Heuristic
            Player(name="PL3", id=2, type='Random'), # CountingHeuristic
            Player(name="PL3", id=3, type='Heuristic') # RL
        ])
        # Convert deque back to list when passing to Golf
        golf = Golf(players=list(players), deck_type="French", verbose=verbose)
        game_results = play_game(golf, game_num, hole, Q, model, shuffle=shuffle)
        ledger.extend(game_results)
        # Rotate players after each turn
        players.rotate(-1)
    
    # Print game statistics
    game_result_df = pd.DataFrame.from_dict(ledger)
    res = game_result_df.groupby(["player_id","game"])['score'].sum().reset_index().groupby("player_id")['score'].mean()

    res['size_Q'] = len(Q)
    #res['x_reward'] = game_result_df.groupby(["player_id","game"])['reward'].sum().reset_index().groupby("player_id")['reward'].mean()
    all_game_results.append(res)
    print(f"Game {game_num}: result")
    print(res)
    print(f"RL Q-table size: {len(Q)}")
    print("="*20)


# print all game results, average score per player and standard deviation
# print title
print("Average score per player and standard deviation")
#print(all_game_results)
all_game_results_df = pd.DataFrame(all_game_results)
#print(all_game_results_df)
#print(all_game_results_df.mean().rename("mean").reset_index().merge(all_game_results_df.std().rename("std").reset_index()))
all_game_results_df.to_csv("all_game_results.csv")
# Save Q-table
with open('q_table.json', 'w') as fp:
    json.dump(Q, fp)

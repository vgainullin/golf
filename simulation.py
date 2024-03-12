from copy import copy, deepcopy
from dataclasses import dataclass
from collections import namedtuple
import random
import numpy as np
import pandas as pd


import collections
from collections.abc import MutableSequence
from collections import namedtuple


Card = namedtuple('Card', ['rank', 'suit'])

class GolfDeck(MutableSequence):
    
    def __init__(self, cards="French"):
        if cards == "French":
            ranks = [str(n) for n in range(2, 11)] + list('JQKA')
            suits = 'spades diamonds clubs hearts'.split()
            cards = [Card(rank, suit) for suit in suits for rank in ranks]
            self._cards = cards
        elif cards == "Blank":
            self._cards = []
        elif cards:
            self._cards = cards

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
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.cards = [
            ["X", "X", "X"],
            ["X", "X", "X"]
            ]
        self.open_cards = [
            ["X", "X", "X"],
            ["X", "X", "X"]
            ]
        self.scores = [
            ["X", "X", "X"],
            ["X", "X", "X"]
            ]
        self.open_ranks = [
            ["X", "X", "X"],
            ["X", "X", "X"]
            ]
        self.holding = None
        self.last_action = None
        self.action_num = 0
        card_to_score = dict(zip([str(n) for n in range(3, 11)] + list("JQKA"), list(range(3, 11))+[10, 10, 0, 1]))
        card_to_score["2"] = -2
        card_to_score["X"] = np.nan
        self.card_to_score = card_to_score
        self.calculate_score()
        self.open_ranks = [
            ["X", "X", "X"],
            ["X", "X", "X"]
            ]

    def card2rank(self, card):
        if card == "X":
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
            if cards[0][i] == cards[1][i] != "X":
                scores[0][i] = 0
                scores[1][i] = 0
            else:
                scores[0][i] = self.card_to_score[cards[0][i]]
                scores[1][i] = self.card_to_score[cards[1][i]]
        scores = np.array(scores)
        score = np.nansum(scores)
        return score, scores

    def calculate_score(self, final=False):
        if final:
            self.open_ranks = self.get_card_ranks(self.cards)
        else:
            self.open_ranks = self.get_card_ranks(self.open_cards)
        self.score, self.scores = self.score_cards(self.open_ranks)

class Golf:
    def __init__(self, players=None):
        self.deck = GolfDeck(cards="French")
        self.discard = GolfDeck(cards="Blank")
        self.face_card = None
        self.players = players
        self.player_num = len(players)
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
        face_card = self.deck.pop()
        self.discard.append(face_card)
        self.face_card = face_card
        
    
    def take_action(self, player_id, action, position=None):
        # step 1: revealed_card = choice (deck.pop(), discard.pop())
        # step 2: if revealed_card != face_card:
        #            -> choice(position, discard), else choice(position)
        if self.last_turn and self.end_game_player_id == player_id:
            self.game_over = True
            print("Game Over")
            return None
        self.players[player_id].last_action = action
        if self.players[player_id].action_num == 0:
            if action == "take_new":
                self.players[player_id].holding = self.deck.pop()
                self.players[player_id].action_num = 1

            elif action == "take_discard":
                self.players[player_id].holding = self.discard.pop()
                self.players[player_id].action_num = 1
            else:
                print("Incorrect action")
                
        elif self.players[player_id].action_num == 1:
            if self.game_over:
                print("ERROR Game ended")
            elif not self.players[player_id].holding:
                print("ERROR, Player should be holding a card")
                # TODO: Handle exit state
                # Throw error
            elif action == "place" and position:
                discard_ = self.players[player_id].cards[position[0]][position[1]]
                self.discard.append(discard_)
                self.players[player_id].cards[position[0]][position[1]] = self.players[player_id].holding
                self.players[player_id].open_cards[position[0]][position[1]] = self.players[player_id].holding
                self.players[player_id].holding = None
                self.players[player_id].action_num = 0
                self.face_card = discard_
            elif action == "discard" and self.players[player_id].holding == self.face_card:
                print("ERROR must place and provide position")
            elif self.players[player_id].holding != self.face_card and action == "discard" and position:
                if self.players[player_id].open_cards[position[0]][position[1]] != "X":
                    print("ERROR Already flipped this card")
                else:
                    self.discard.append(self.players[player_id].holding)
                    self.face_card = self.players[player_id].holding
                    self.players[player_id].holding = None
                    self.players[player_id].open_cards[position[0]][position[1]] = self.players[player_id].cards[position[0]][position[1]]
                    self.players[player_id].action_num = 0
            else:
                print("ERROR No condition met")
        self.players[player_id].calculate_score()
        if not self.game_over:
            if not np.isnan(self.players[player_id].scores).any():
                self.last_turn = True
                self.end_game_player_id = player_id


def get_player_action(game, player_id, rank_cutoff=5):
    if game.players[player_id].action_num == 0:
        rank_of_face_card = game.players[player_id].card2rank(game.face_card)
        # if rank of face card matches one in deck
        rank_match = np.argwhere(game.players[player_id].open_ranks == rank_of_face_card)
        
        if game.players[player_id].card_to_score[game.face_card[0]] < rank_cutoff or rank_match.size > 0:
            return "take_discard", None
        else:
            return "take_new", None
    if game.players[player_id].action_num == 1:
        # calculate optimal placement for card
        # iterate through each possible placement position
        # calculate score
        # choose min score
        current_score = game.players[player_id].score
        min_score = 99
        for row in range(2):
            for c in range(3):
                player_cards_copy = deepcopy(game.players[player_id].open_cards)
                player_cards_copy[row][c] = game.players[player_id].holding
                player_card_ranks = game.players[player_id].get_card_ranks(player_cards_copy)
                score, scores = game.players[player_id].score_cards(player_card_ranks)
                #print((row, c), score)
                if score < min_score:
                    min_score = score
                    opt_pos = (row, c)
                    upd_score = score
        # If found a place that reduces current score by rank_cutoff
        golf.players[player_id].calculate_score()
        available_pos_to_place = np.argwhere(np.isnan(golf.players[player_id].scores))
        if len(available_pos_to_place) > 0:
            can_place = True
        else:
            can_place = False
        if upd_score <= (current_score - rank_cutoff):
            action = ("place", opt_pos)
        
        elif can_place and (upd_score - current_score) < rank_cutoff:
            # card value is less than rank, place it somewhere
            action = ("place", tuple(available_pos_to_place[0]))
        elif can_place:
            # discard and flip one of the cards instead
            action = ("discard", tuple(available_pos_to_place[0]))

        return action


def take_turn(player_id, game, rank_cutoff=5):
    if game.game_over:
        print("GAME OVER")
    else:
        for i in range(2):
            action, pos = get_player_action(deepcopy(game), player_id, rank_cutoff)
            game.take_action(player_id=player_id, action=action, position=pos)

max_num_rounds = 100

ledger = []
rank_cutoff = 6
for game_num in range(1):
    for hole in range(10):
        players = [Player(name="PL1", id=0), Player(name="PL2", id=1), Player(name="PL3", id=2)]
        golf = Golf(players=players)
        golf.shuffle()
        golf.deal()
        
        round_num = 0
        while round_num < max_num_rounds and not golf.last_turn:
            for player_id in range(3):
                if not golf.last_turn and golf.end_game_player_id != player_id:
                    take_turn(player_id, golf, rank_cutoff)
            round_num += 1
                    
        for player in golf.players:
            
            player.calculate_score(final=True)
            ledger.append(dict(player_id=player.id, score=player.score, hole=hole, game=game_num))
game_result_df = pd.DataFrame.from_dict(ledger)
res = game_result_df.groupby(["player_id","game"])['score'].sum().reset_index().groupby("player_id")['score'].mean()
print(res)

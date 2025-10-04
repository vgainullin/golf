# tests/test_simulation.py
import json
import random

import numpy as np
import pytest

import src.simulation as simulation_mod
from src.simulation import Player, GolfDeck, Card

@pytest.fixture
def player():
    """Create a test player fixture."""
    return Player(name="TestPlayer", id=0)

@pytest.fixture
def deck():
    """Create a test deck fixture."""
    return GolfDeck()

def test_initial_score(player):
    """Test that the initial score is set correctly."""
    assert player.score == 0

def test_card2rank(player):
    """Test the card2rank method."""
    card = Card(rank='A', suit='spades')
    assert player.card2rank(card) == 'A'
    assert player.card2rank(None) is None
    assert player.card2rank("?") == "?"

def test_deck_creation(deck):
    """Test deck creation."""
    assert len(deck) == 52
    assert isinstance(deck[0], Card)

def test_deck_types():
    """Test different deck types."""
    french_deck = GolfDeck("French")
    double_deck = GolfDeck("2xFrench")
    blank_deck = GolfDeck("Blank")
    
    assert len(french_deck) == 52
    assert len(double_deck) == 104
    assert len(blank_deck) == 0
    
    with pytest.raises(ValueError):
        GolfDeck("InvalidType")

def test_card_creation():
    """Test card creation and properties."""
    card = Card(rank='K', suit='hearts')
    assert card.rank == 'K'
    assert card.suit == 'hearts'
    print(str(card) == 'K hearts')

def test_deck_operations(deck):
    """Test deck operations like drawing and inserting cards."""
    initial_length = len(deck)
    card = deck[0]
    
    # Test deletion
    del deck[0]
    assert len(deck) == initial_length - 1
    
    # Test insertion
    deck.insert(0, card)
    assert len(deck) == initial_length
    assert deck[0] == card

def test_card2index(deck):
    """Test card to index conversion."""
    # Test regular card
    card = Card(rank='A', suit='spades')
    index = deck.card2index(card)
    assert isinstance(index, int)
    assert 0 <= index < 52
    
    # Test unknown card
    unknown_card = Card(rank='?', suit='spades')
    unknown_index = deck.card2index(unknown_card)
    assert unknown_index == 53  # Outside normal range

def test_player_score_calculation(player):
    """Test player score calculation with different card combinations."""
    # Set up some test cards
    player.cards = [
        Card(rank='K', suit='hearts'),
        Card(rank='K', suit='spades')
    ]
    player.calculate_score()
    assert player.score == 0  # Matching pairs should score 0
    
    player.cards = [
        Card(rank='K', suit='hearts'),
        Card(rank='Q', suit='spades')
    ]
    player.calculate_score()
    assert player.score  == 0

def test_player_game_state(player):
    """Test player game state gathering."""
    player.cards = [
        Card(rank='K', suit='hearts'),
        Card(rank='Q', suit='spades')
    ]
    player.open_cards = [
        [Card(rank='A', suit='diamonds')],
        [Card(rank='2', suit='clubs')]
    ]
    player.gather_game_state()
    assert isinstance(player.game_state, str)
    assert len(player.game_state) > 0

def test_deck_shuffling(deck):
    """Test deck shuffling."""
    original_order = deck._cards.copy()
    # only game can shuffle decks
    # deck.shuffle()
    random.shuffle(deck)
    assert len(deck) == len(original_order)
    assert deck._cards != original_order  # Note: there's a tiny chance this could fail randomly

@pytest.mark.parametrize("rank,suit", [
    ('A', 'spades'),
    ('K', 'hearts'),
    ('2', 'diamonds'),
    ('X', 'clubs'),
])
def test_valid_cards(rank, suit):
    """Test creation of various valid cards."""
    card = Card(rank=rank, suit=suit)
    assert card.rank == rank
    assert card.suit == suit

def test_player_holding_card(player):
    """Test player's holding card functionality."""
    card = Card(rank='A', suit='spades')
    player.holding = card
    assert player.holding == card
    player.holding = None
    assert player.holding is None

def test_encode_golf_tensor_simple():
    """Test Golf.encode_golf_tensor for a simple deterministic state."""
    from src.simulation import Golf, Player, Card
    # 2 players, each with known cards, no holding, deck/discard/face empty
    players = [
        Player(name="P0", id=0),
        Player(name="P1", id=1)
    ]
    # Use the Golf.pos_index static method for position mapping

    players[0].cards = [
        [Card(rank='2', suit='spades'), "?", "?"],
        ["?", "?", "?"]
    ]
    players[1].cards = [
        ["?", "?", "?"],
        ["?", "?", Card(rank='3', suit='hearts')]
    ]
    golf = Golf(players=players)
    golf.deck = GolfDeck(cards="Blank")  # No cards in deck
    golf.discard = GolfDeck(cards="Blank")
    golf.face_card = None
    tensor = golf.encode_golf_tensor()
    # Check shape
    assert tensor.shape == (13, 2*7+3, 4)
    # 2 of spades should be at [0,0,0] (rank 2, player 0 slot 0, spades)
    assert tensor[golf.deck.ranks.index('2'), Golf.pos_index(0, 0, 0), golf.deck.suits.index('spades')] == 1
    # 3 of hearts should be at [1,13,3] (rank 3, player 1 slot 6, hearts)
    assert tensor[golf.deck.ranks.index('3'), Golf.pos_index(1, 1, 2), golf.deck.suits.index('hearts')] == 1
    # All other entries should be 0
    assert tensor.sum() == 2


def test_tensor_transition_logger_roundtrip(tmp_path):
    """TensorTransitionLogger should persist tensors and metadata."""
    logger = simulation_mod.TensorTransitionLogger(tmp_path)
    state = np.zeros((13, 10, 4), dtype=np.int8)
    next_state = state.copy()
    logger.log(
        state=state,
        next_state=next_state,
        reward=1.5,
        done=False,
        metadata={
            "game": 0,
            "hole": 1,
            "round": 0,
            "player_id": 0,
            "action_num": 0,
            "action": 1,
            "position": None,
        },
    )

    logger.save(prefix="test_log")
    archive = tmp_path / "test_log.npz"
    metadata_path = tmp_path / "test_log.json"

    assert archive.exists()
    assert metadata_path.exists()

    payload = np.load(archive)
    assert payload["states"].shape == (1, 13, 10, 4)
    assert payload["next_states"].shape == (1, 13, 10, 4)
    assert payload["rewards"][0] == pytest.approx(1.5)
    assert payload["dones"][0] == False

    metadata = json.loads(metadata_path.read_text())
    assert metadata[0]["game"] == 0
    assert metadata[0]["position"] is None
    assert metadata[0]["done"] is False


def test_play_game_logs_tensor_transitions(tmp_path):
    """play_game should log tensor transitions when a logger is provided."""
    original_verbose = simulation_mod.verbose
    simulation_mod.verbose = False
    random.seed(0)

    players = [
        Player(name="P0", id=0, type='Heuristic'),
        Player(name="P1", id=1, type='Heuristic'),
    ]
    golf = simulation_mod.Golf(players=players, deck_type="French", verbose=False)
    logger = simulation_mod.TensorTransitionLogger(tmp_path)

    Q = {}
    model = simulation_mod.QTransformer()

    try:
        results = simulation_mod.play_game(
            golf,
            game_num=0,
            hole=1,
            Q=Q,
            model=model,
            shuffle=False,
            transition_logger=logger,
        )
    finally:
        simulation_mod.verbose = original_verbose

    assert results
    metadata_entries = tuple(logger.metadata)
    assert metadata_entries

    logger.save(prefix="play_game")
    archive = tmp_path / "play_game.npz"
    payload = np.load(archive)
    assert payload["states"].shape[0] == len(metadata_entries)
    assert payload["states"].shape[1:] == (13, len(players) * 7 + 3, 4)

    metadata = json.loads((tmp_path / "play_game.json").read_text())
    assert len(metadata) == len(metadata_entries)

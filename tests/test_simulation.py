# tests/test_simulation.py
import pytest
import random
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
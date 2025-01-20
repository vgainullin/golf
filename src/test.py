import pygame
import random
from collections import namedtuple
from collections.abc import MutableSequence

GolfCard = namedtuple("Card", ["rank", "suit"])


class GolfDeck(MutableSequence):
    def __init__(self, cards="French"):
        if cards == "French":
            ranks = [str(n) for n in range(2, 11)] + list("JQKA")
            suits = "spades diamonds clubs hearts".split()
            cards = [GolfCard(rank, suit) for suit in suits for rank in ranks]
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


class Card(object):
    def __init__(self, x, y, text):
        self.font = pygame.font.SysFont("Arial", 25)
        self.text = text
        self.x = x
        self.y = y
        self.color = (255, 0, 0)  # Text color (red in this example)
        self.text_surface = self.font.render(self.text, True, self.color)
        self.card_rect = pygame.Rect(self.x, self.y, CARD_SIZE[0], CARD_SIZE[1])
        self.card_rect_h = pygame.Rect(
            self.x + 20, self.y + 20, CARD_SIZE[0] + 25, CARD_SIZE[1] + 25
        )
        self.dragging = False  # Flag to indicate if the card is being dragged
        self.in_c1_area = False  # Flag to indicate if the card is in a valid area
        self.in_discard_area = False
        self.is_highlighted = False
        self.showing_face = True

    def draw(self, screen):
        # Draw the rectangle
        if self.is_highlighted:
            pygame.draw.rect(screen, (0, 150, 255), self.card_rect)
        else:
            pygame.draw.rect(screen, (0, 0, 255), self.card_rect)

        if self.showing_face:
            # Center the text within the rectangle
            text_rect = self.text_surface.get_rect(center=self.card_rect.center)
            # Draw the text on the screen
            screen.blit(self.text_surface, text_rect)

    def start_drag(self, mouse_x, mouse_y):
        # Check if the mouse click is within the card's bounds
        if self.card_rect.collidepoint(mouse_x, mouse_y):
            self.dragging = True
            self.offset_x = mouse_x - self.x
            self.offset_y = mouse_y - self.y

    def end_drag(self):
        self.dragging = False

    def update_position(self, pos):
        self.x, self.y = pos
        self.card_rect.topleft = pos

    def drag(self, mouse_x, mouse_y):
        if self.dragging:
            new_x = mouse_x - self.offset_x
            new_y = mouse_y - self.offset_y

            # Update the card's position
            self.x = new_x
            self.y = new_y
            self.card_rect.topleft = (self.x, self.y)

    def flip(self):
        if self.showing_face:
            self.showing_face = False
        else:
            self.showing_face = True


class Button(object):
    def __init__(self, x, y, width, height, text, click_action):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.click_action = click_action
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.SysFont("Arial", 20)
        self.visible = False

    def show(self, position):
        self.x = position[0] + 20
        self.y = position[1]
        self.rect.topleft = (self.x, self.y)
        self.visible = True

    def hide(self):
        self.visible = False

    def draw(self, screen):
        if self.visible:
            pygame.draw.rect(screen, (0, 0, 0), self.rect)
            text_surface = self.font.render(self.text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)


# Constants
CARD_SIZE = (100, 150)
DISCARD_PILE = pygame.Rect(600, 400, CARD_SIZE[0], CARD_SIZE[1])
DRAW_AREA = pygame.Rect(600, 40, CARD_SIZE[0], CARD_SIZE[1])

# Create card placement area
CARD_AREA_1 = pygame.Rect(100, 400, CARD_SIZE[0] * 2, CARD_SIZE[1] * 2)

# Initialize Pygame
pygame.init()

# Create the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Card Drawing")

# Create Card1 with initial position in CARD_AREA_1
deck = GolfDeck()
random.shuffle(deck)
card = deck[-1]
Card1 = Card(150, 450, card[0])


# Create a flip button
flip_button = Button(700, 100, 80, 30, "Flip", lambda: Card1.flip())

# Variables to track card movement
card_being_dragged = None
original_position = None

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Left mouse button down
            if Card1.card_rect.collidepoint(event.pos):
                card_being_dragged = Card1
                original_position = (Card1.x, Card1.y)
                card_being_dragged.start_drag(*event.pos)
        elif (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and card_being_dragged
        ):
            # Left mouse button down
            if card_being_dragged.in_discard_area:
                card_being_dragged.update_position(DISCARD_PILE.topleft)

            elif card_being_dragged.in_c1_area:
                card_being_dragged.update_position(CARD_AREA_1.topleft)

            else:
                card_being_dragged.update_position(original_position)
            card_being_dragged.is_highlighted = False
            card_being_dragged.end_drag()
            card_being_dragged = None
        elif (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 3
            and not card_being_dragged
            and Card1.card_rect.collidepoint(event.pos)
            and not flip_button.visible
        ):
            # Right mouse button down
            # flip_button.click_action()
            flip_button.visible = True
            flip_button.show(position=event.pos)
        elif (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and flip_button.rect.collidepoint(event.pos)
        ):
            # Right mouse button down
            print("Clicked")
            flip_button.click_action()
            flip_button.visible = False

    if card_being_dragged and card_being_dragged.dragging:
        card_being_dragged.drag(*pygame.mouse.get_pos())
        if card_being_dragged.card_rect.colliderect(DISCARD_PILE):
            card_being_dragged.in_discard_area = True
            card_being_dragged.is_highlighted = True
        elif card_being_dragged.card_rect.colliderect(CARD_AREA_1):
            card_being_dragged.in_c1_area = True
            card_being_dragged.is_highlighted = True
        else:
            card_being_dragged.in_c1_area = False
            card_being_dragged.in_discard_area = False
            card_being_dragged.is_highlighted = False

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw black rectangles for card placement area and discard pile
    pygame.draw.rect(screen, (0, 0, 0), CARD_AREA_1, 2)  # Outline
    pygame.draw.rect(screen, (0, 0, 0), DISCARD_PILE, 2)  # Outline
    pygame.draw.rect(screen, (0, 0, 0), DRAW_AREA, 2)  # Outline

    # Draw Card1

    Card1.draw(screen)

    if flip_button.visible:
        flip_button.draw(screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

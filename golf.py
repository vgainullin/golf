import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 240)

RED = (255, 0, 0)
GREEN = (0, 81, 44)
BLUE = (0, 0, 255)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CARD_SIZE = (100, 120)


pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen_rect = screen.get_rect()

font = pg.font.Font(None, 46)
color = pg.Color("sienna1")


class Card(object):
    def __init__(self, x, y, text):
        self.font = pygame.font.SysFont("Arial", 25)
        self.text = text
        self.x = x
        self.y = y
        self.color = (255, 0, 0)  # Text color (red in this example)
        self.text_surface = self.font.render(self.text, True, self.color)
        self.card_rect = pygame.Rect(self.x, self.y, CARD_SIZE[0], CARD_SIZE[1])

    def draw(self, screen):
        # Draw the rectangle
        pygame.draw.rect(screen, (0, 0, 255), self.card_rect)

        # Center the text within the rectangle
        text_rect = self.text_surface.get_rect(center=self.card_rect.center)

        # Draw the text on the screen
        screen.blit(self.text_surface, text_rect)


cards = []

for x in range(6):
    card = Card()
    card_rect = pygame.Rect(
        x * (CARD_SIZE[0] + 5), CARD_SIZE[0], CARD_SIZE[0], CARD_SIZE[1]
    )

    cards.append(card_rect)

selected = None

# --- mainloop ---

clock = pygame.time.Clock()
is_running = True

while is_running:
    # --- events ---

    for event in pygame.event.get():
        # --- global events ---

        if event.type == pygame.QUIT:
            is_running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                is_running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for i, r in enumerate(cards):
                    if r.collidepoint(event.pos):
                        selected = i
                        selected_offset_x = r.x - event.pos[0]
                        selected_offset_y = r.y - event.pos[1]

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                selected = None

        elif event.type == pygame.MOUSEMOTION:
            if selected is not None:  # selected can be `0` so `is not None` is required
                # move object
                cards[selected].x = event.pos[0] + selected_offset_x
                cards[selected].y = event.pos[1] + selected_offset_y

    screen.fill(GREEN)

    for r in cards:
        pygame.draw.rect(screen, WHITE, r)

    pygame.display.update()

    # --- FPS ---

    clock.tick(25)

# --- the end ---

pygame.quit()

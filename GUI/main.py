# GUI/main_gui.py
import pygame
import sys
from hex_utils import (
    draw_hex_grid, get_board_dimensions,
    COLOR_BACKGROUND, HEX_SIZE
)

# It's good practice to import from your project's constants
# Assuming constants.py is in the parent directory
sys.path.append('..') # Add parent directory to sys.path
from constants import VALID_HEXES # Your set of (q,r) tuples

# --- Pygame Setup ---
pygame.init()
pygame.font.init() # Initialize font module

# Screen dimensions (will be adjusted)
INFO_PANEL_HEIGHT = 150 # Space for scores, messages, etc.
BOARD_SPACING = 50    # Space between the two boards
SIDE_PADDING = 50     # Padding on the left/right of boards

# --- Game Data (Placeholders for now) ---
# We'll eventually get this from HarmoniesGameState
player0_board_data = { # Example: (q,r): [tile_stack]
    (0, 0): ["wood", "plant"],
    (1, -1): ["stone"],
    (-1, 1): ["water"]
}
player1_board_data = {
    (0, -2): ["building"],
    (0, 1): ["field", "field"]
}

# --- Calculate Screen Dimensions ---
# Get dimensions for a single board
single_board_width_approx, single_board_height_approx = get_board_dimensions(VALID_HEXES)

# Calculate total screen width and height
SCREEN_WIDTH = (SIDE_PADDING * 2) + (single_board_width_approx * 2) + BOARD_SPACING
SCREEN_HEIGHT = single_board_height_approx + INFO_PANEL_HEIGHT + (SIDE_PADDING * 2) # Add top/bottom padding

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Harmonies AI GUI")
clock = pygame.time.Clock()
game_font = pygame.font.SysFont("Arial", 18) # A basic font

# --- Calculate Board Origins ---
# Origin for Player 0's board (left)
# Centering the board in its allocated horizontal space
board0_origin_x = SIDE_PADDING + single_board_width_approx // 2
board0_origin_y = SIDE_PADDING + single_board_height_approx // 2

# Origin for Player 1's board (right)
board1_origin_x = board0_origin_x + single_board_width_approx + BOARD_SPACING
board1_origin_y = board0_origin_y # Same y-level

# --- Main Game Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # --- Drawing ---
    screen.fill(COLOR_BACKGROUND)

    # Draw Player 0's board
    title0 = game_font.render("Player 0 Board", True, (0,0,0))
    screen.blit(title0, (board0_origin_x - title0.get_width()//2, board0_origin_y - single_board_height_approx//2 - 25))
    draw_hex_grid(screen, board0_origin_x, board0_origin_y, VALID_HEXES, player0_board_data, game_font)

    # Draw Player 1's board (AI)
    title1 = game_font.render("Player 1 Board (AI)", True, (0,0,0))
    screen.blit(title1, (board1_origin_x - title1.get_width()//2, board1_origin_y - single_board_height_approx//2 - 25))
    draw_hex_grid(screen, board1_origin_x, board1_origin_y, VALID_HEXES, player1_board_data, game_font)

    # --- Info Panel (Placeholder) ---
    info_panel_rect = pygame.Rect(0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, (220, 210, 200), info_panel_rect) # Slightly different background
    info_text = game_font.render("Game Info / Stats Area", True, (0,0,0))
    screen.blit(info_text, (info_panel_rect.x + 10, info_panel_rect.y + 10))


    pygame.display.flip() # Update the full screen
    clock.tick(30)        # Limit to 30 FPS

pygame.quit()
sys.exit()
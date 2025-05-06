# GUI/main_gui.py
import pygame
import sys

# --- CHOOSE YOUR HEX ORIENTATION ---
# Option 1: Pointy-Topped (vertical stacking)
from hex_utils import (
    draw_hex_grid_pointy_topped as draw_hex_grid, # Alias for generic use
    get_board_dimensions_pointy_topped as get_board_dimensions,
    COLOR_BACKGROUND, HEX_SIZE, HEX_HEIGHT, HEX_WIDTH
)
# Option 2: Flat-Topped (horizontal stacking) - uncomment these and comment above if preferred
# from hex_utils import (
#     draw_hex_grid_flat_topped as draw_hex_grid,
#     get_board_dimensions_flat_topped as get_board_dimensions,
#     COLOR_BACKGROUND, HEX_SIZE, HEX_HEIGHT, HEX_WIDTH
# )

sys.path.append('..') # Add parent directory to sys.path
from constants import VALID_HEXES, TILE_TYPES # Your set of (q,r) tuples and tile types
from harmonies_engine import HarmoniesGameState # Import your game engine

# --- Pygame Setup ---
pygame.init()
pygame.font.init()

# Screen dimensions
INFO_PANEL_HEIGHT = 200 # Increased for more info
BOARD_SPACING = 70
SIDE_PADDING = 50
TOP_PADDING = 50 # Padding above boards

# --- Game Initialization ---
game_state = HarmoniesGameState() # Create an instance of your game
human_player_id = 0 # Let's assume human is Player 0 for now
ai_player_id = 1    # AI will be Player 1

# --- Calculate Screen Dimensions (based on chosen hex orientation) ---
single_board_width_approx, single_board_height_approx = get_board_dimensions(VALID_HEXES)
# Add some padding to the calculated dimensions
single_board_width_approx += int(HEX_WIDTH * 0.5)
single_board_height_approx += int(HEX_HEIGHT * 0.5)

SCREEN_WIDTH = (SIDE_PADDING * 2) + (single_board_width_approx * 2) + BOARD_SPACING
SCREEN_HEIGHT = TOP_PADDING + single_board_height_approx + INFO_PANEL_HEIGHT + SIDE_PADDING # Added TOP_PADDING

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Harmonies GUI")
clock = pygame.time.Clock()
game_font_small = pygame.font.SysFont("Arial", 16)
game_font_medium = pygame.font.SysFont("Arial", 20)
game_font_large = pygame.font.SysFont("Arial", 24)


# --- Calculate Board Origins ---
board0_visual_center_x = SIDE_PADDING + single_board_width_approx // 2
board0_visual_center_y = TOP_PADDING + single_board_height_approx // 2

board1_visual_center_x = board0_visual_center_x + single_board_width_approx + BOARD_SPACING
board1_visual_center_y = board0_visual_center_y

# --- Helper to draw multiline text ---
def draw_text_multiline(surface, text, pos, font, color, max_width=None):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    x, y = pos
    line_spacing = font.get_linesize()

    for line_words in words:
        current_line_width = 0
        line_start_x = x
        for i, word in enumerate(line_words):
            word_surface = font.render(word, True, color)
            word_width, word_height = word_surface.get_size()
            if max_width and current_line_width + word_width >= max_width:
                y += line_spacing  # New line
                current_line_width = 0
                line_start_x = x # Reset x for new line
            surface.blit(word_surface, (line_start_x + current_line_width, y))
            current_line_width += word_width + space
        y += line_spacing # Next line after processing all words in current original line

# --- Main Game Loop ---
running = True
action_prompt = "" # To display what action the human needs to take

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # --- Basic Human Input (Console for now) ---
            if game_state.get_current_player() == human_player_id and not game_state.is_game_over():
                if event.key == pygame.K_RETURN: # Use Enter key to trigger console input
                    legal_moves = game_state.get_legal_moves()
                    if not legal_moves:
                        print("No legal moves for you!")
                        action_prompt = "No legal moves!"
                    else:
                        print("\n--- Your Turn ---")
                        action_prompt = f"Phase: {game_state.turn_phase}. Enter move in console."
                        pygame.display.set_caption(f"Harmonies GUI - {action_prompt}") # Update window title

                        if game_state.turn_phase == "choose_pile":
                            print("Available Piles:")
                            for i, pile_idx in enumerate(legal_moves): # legal_moves are pile indices
                                if 0 <= pile_idx < len(game_state.available_piles):
                                    print(f"  {i+1}: Pile {pile_idx} ({'/'.join(game_state.available_piles[pile_idx])})")
                            try:
                                choice_str = input(f"Choose pile (1-{len(legal_moves)}): ")
                                choice = int(choice_str) - 1
                                if 0 <= choice < len(legal_moves):
                                    chosen_action = legal_moves[choice]
                                    game_state = game_state.apply_move(chosen_action)
                                    action_prompt = "" # Clear prompt
                                else: action_prompt = "Invalid pile choice."
                            except ValueError:
                                action_prompt = "Invalid input. Enter a number."
                        elif game_state.turn_phase.startswith("place_tile"):
                            print(f"Your hand: {game_state.tiles_in_hand}")
                            print("Legal placements:")
                            for i, (tile_type, coord) in enumerate(legal_moves): # type: ignore # legal_moves are (type, coord)
                                print(f"  {i+1}: Place {tile_type.upper()} at {coord}")
                            try:
                                choice_str = input(f"Choose placement (1-{len(legal_moves)}): ")
                                choice = int(choice_str) - 1
                                if 0 <= choice < len(legal_moves):
                                    chosen_action = legal_moves[choice]
                                    game_state = game_state.apply_move(chosen_action)
                                    action_prompt = "" # Clear prompt
                                else: action_prompt = "Invalid placement choice."
                            except ValueError:
                                action_prompt = "Invalid input. Enter a number."
                        pygame.display.set_caption("Harmonies GUI") # Reset title

    # --- Game Logic (AI Turn - Placeholder for now) ---
    if game_state.get_current_player() == ai_player_id and not game_state.is_game_over():
        action_prompt = "AI is thinking..."
        # TODO: Implement AI move logic here (e.g., call MCTS or Greedy)
        # For now, let's make a dummy move or pass if no legal moves
        legal_ai_moves = game_state.get_legal_moves()
        if legal_ai_moves:
            # ai_chosen_action = random.choice(legal_ai_moves) # Placeholder: AI picks randomly
            # game_state = game_state.apply_move(ai_chosen_action)
            action_prompt = "AI made a move (dummy)" # Update after actual AI move
            print(f"AI (dummy) would consider moves: {legal_ai_moves[:3]}")
            # We will replace this with actual AI call and move application
            # For now, to prevent infinite loop, let's not have AI move automatically
            # and wait for human to trigger next step or manually advance turn.
            pass # AI doesn't move yet to allow focus on display
        else:
            action_prompt = "AI has no legal moves."
            # Handle end of AI turn logic (e.g., if it means game ends or phase changes)
            if game_state.turn_phase.startswith("place_tile") and not game_state.tiles_in_hand:
                if game_state.turn_phase == "place_tile_1": game_state.turn_phase = "place_tile_2"
                elif game_state.turn_phase == "place_tile_2": game_state.turn_phase = "place_tile_3"
                elif game_state.turn_phase == "place_tile_3": game_state._end_turn_actions()


    # --- Drawing ---
    screen.fill(COLOR_BACKGROUND)

    # Draw Player 0's board (Human)
    title0_str = f"Player {human_player_id} Board (You)"
    if game_state.get_current_player() == human_player_id: title0_str += " *" # Indicate current turn
    title0 = game_font_medium.render(title0_str, True, (0,0,0))
    screen.blit(title0, (board0_visual_center_x - title0.get_width()//2,
                        board0_visual_center_y - single_board_height_approx//2 - 30))
    draw_hex_grid(screen, board0_visual_center_x, board0_visual_center_y,
                  VALID_HEXES, game_state.player_boards[human_player_id], game_font_small)

    # Draw Player 1's board (AI)
    title1_str = f"Player {ai_player_id} Board (AI)"
    if game_state.get_current_player() == ai_player_id: title1_str += " *"
    title1 = game_font_medium.render(title1_str, True, (0,0,0))
    screen.blit(title1, (board1_visual_center_x - title1.get_width()//2,
                        board1_visual_center_y - single_board_height_approx//2 - 30))
    draw_hex_grid(screen, board1_visual_center_x, board1_visual_center_y,
                  VALID_HEXES, game_state.player_boards[ai_player_id], game_font_small)

    # --- Info Panel ---
    info_panel_rect = pygame.Rect(0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, (210, 200, 190), info_panel_rect) # Slightly different background
    pygame.draw.line(screen, (150,150,150), (info_panel_rect.left, info_panel_rect.top), (info_panel_rect.right, info_panel_rect.top), 2)


    # Display Game Info
    info_y_start = info_panel_rect.y + 10
    line_height = game_font_small.get_linesize()
    info_x_col1 = info_panel_rect.x + 20
    info_x_col2 = info_panel_rect.centerx + 10

    # Column 1 Info
    current_player_text = f"Current Player: {game_state.get_current_player()}"
    phase_text = f"Turn Phase: {game_state.turn_phase}"
    scores_text = f"Scores: P0: {game_state.final_scores[0]}, P1: {game_state.final_scores[1]}" # Will be 0 until game over
    
    screen.blit(game_font_medium.render(current_player_text, True, (0,0,0)), (info_x_col1, info_y_start))
    screen.blit(game_font_small.render(phase_text, True, (0,0,0)), (info_x_col1, info_y_start + line_height * 1.5))
    screen.blit(game_font_small.render(scores_text, True, (0,0,0)), (info_x_col1, info_y_start + line_height * 3))

    if action_prompt: # Display any prompts for human
        prompt_surface = game_font_small.render(action_prompt, True, (200,0,0))
        screen.blit(prompt_surface, (info_x_col1, info_y_start + line_height * 4.5))


    # Column 2 Info (Piles, Hand, Bag)
    available_piles_str = "Available Piles:\n"
    if game_state.available_piles:
        for i, pile in enumerate(game_state.available_piles):
            available_piles_str += f"  {i}: {'/'.join(pile)}\n"
    else:
        available_piles_str += "  (None)\n"
    draw_text_multiline(screen, available_piles_str, (info_x_col2, info_y_start), game_font_small, (0,0,0), max_width=SCREEN_WIDTH // 2 - 30)

    hand_text_y = info_y_start + line_height * (available_piles_str.count('\n') + 0.5) # Position below piles
    if game_state.turn_phase.startswith("place_tile") and game_state.get_current_player() == human_player_id:
        hand_str = f"Your Hand: {', '.join(game_state.tiles_in_hand) if game_state.tiles_in_hand else '(Empty)'}"
        screen.blit(game_font_small.render(hand_str, True, (0,0,0)), (info_x_col2, hand_text_y))
        hand_text_y += line_height

    bag_str = "Tile Bag:\n"
    for tile_type in TILE_TYPES:
        count = game_state.tile_bag.get(tile_type, 0)
        if count > 0 : bag_str += f"  {tile_type.capitalize()}: {count}\n"
    draw_text_multiline(screen, bag_str, (info_x_col2, hand_text_y + line_height * 0.5), game_font_small, (0,0,0), max_width=SCREEN_WIDTH // 2 - 30)


    if game_state.is_game_over():
        winner_text = ""
        outcome = game_state.get_game_outcome()
        if outcome == 1: winner_text = f"Player 0 WINS!"
        elif outcome == -1: winner_text = f"Player 1 WINS!"
        else: winner_text = "It's a DRAW!"
        winner_surf = game_font_large.render(winner_text, True, (200, 0, 0))
        screen.blit(winner_surf, (SCREEN_WIDTH // 2 - winner_surf.get_width() // 2, TOP_PADDING // 2))


    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
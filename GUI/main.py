import pygame
import sys
import os

# --- CHOOSE YOUR HEX ORIENTATION ---
# Make sure hex_utils.py is in the same GUI/ directory or adjust path
# Option 1: Pointy-Topped (vertical stacking)
from hex_utils import (
    draw_hex_grid_pointy_topped as draw_hex_grid, # Alias for generic use
    get_board_dimensions_pointy_topped as get_board_dimensions,
    COLOR_BACKGROUND, HEX_SIZE, HEX_HEIGHT, HEX_WIDTH # Make sure these are from pointy-topped context
)
# Option 2: Flat-Topped (horizontal stacking) - uncomment these and comment above if preferred
# from hex_utils import (
#     draw_hex_grid_flat_topped as draw_hex_grid,
#     get_board_dimensions_flat_topped as get_board_dimensions,
#     COLOR_BACKGROUND, HEX_SIZE, HEX_HEIGHT, HEX_WIDTH
# )

# Add parent directory to sys.path to allow imports of your game modules
# This assumes main_gui.py is in GUI/ and your game files are in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os # Import os here if not already

from constants import VALID_HEXES, TILE_TYPES
from harmonies_engine import HarmoniesGameState # Import your game engine

# --- Pygame Setup ---
pygame.init()
pygame.font.init() # Initialize font module

# Screen dimensions
INFO_PANEL_HEIGHT = 200
BOARD_SPACING = 70
SIDE_PADDING = 50
TOP_PADDING = 50

# Game Fonts
game_font_small = pygame.font.SysFont("Arial", 16)
game_font_medium = pygame.font.SysFont("Arial", 20)
game_font_large = pygame.font.SysFont("Arial", 24)

clock = pygame.time.Clock() # Define clock globally for TextInputBox

# --- TextInputBox Class ---
class TextInputBox:
    def __init__(self, x, y, w, h, font, initial_text="", text_color=(0,0,0), box_color_inactive=(200,200,200), box_color_active=(230,230,230), border_color=(0,0,0), prompt=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.text_color = text_color
        self.box_color_inactive = box_color_inactive
        self.box_color_active = box_color_active
        self.border_color = border_color
        self.user_text = initial_text
        self.prompt = prompt
        self.prompt_surface = self.font.render(self.prompt, True, self.text_color)
        self.is_active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.text_surface = self.font.render(self.user_text, True, self.text_color)

    def handle_event(self, event):
        submitted_text = None # Default to no submission
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_active = True
            else:
                self.is_active = False # Deactivate if clicked outside
        
        if self.is_active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                submitted_text = self.user_text
                # self.user_text = "" # Optionally clear after submit
                # self.is_active = False # Optionally deactivate after submit
            elif event.key == pygame.K_BACKSPACE:
                self.user_text = self.user_text[:-1]
            else:
                # Allow numbers and basic characters, can be expanded
                if event.unicode.isalnum() or event.unicode in ['.', '-', ' ']:
                    self.user_text += event.unicode
            self.text_surface = self.font.render(self.user_text, True, self.text_color)
        return submitted_text

    def update(self):
        # Cursor blinking
        if self.is_active:
            self.cursor_timer += clock.get_time() 
            if self.cursor_timer >= 500: 
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0
        else:
            self.cursor_visible = False # No cursor if not active

    def draw(self, surface):
        current_box_color = self.box_color_active if self.is_active else self.box_color_inactive
        pygame.draw.rect(surface, current_box_color, self.rect)
        pygame.draw.rect(surface, self.border_color, self.rect, 2)

        prompt_render_x = self.rect.x - self.prompt_surface.get_width() - 5
        prompt_render_y = self.rect.y + (self.rect.height - self.prompt_surface.get_height()) // 2
        surface.blit(self.prompt_surface, (prompt_render_x, prompt_render_y))
        
        text_render_x = self.rect.x + 5
        text_render_y = self.rect.y + (self.rect.height - self.text_surface.get_height()) // 2
        surface.blit(self.text_surface, (text_render_x, text_render_y))

        if self.is_active and self.cursor_visible:
            cursor_x = text_render_x + self.text_surface.get_width() + 2
            cursor_y_start = self.rect.y + 5
            cursor_y_end = self.rect.y + self.rect.height - 5
            pygame.draw.line(surface, self.text_color, (cursor_x, cursor_y_start), (cursor_x, cursor_y_end), 2)

# --- Helper to draw multiline text ---
def draw_text_multiline(surface, text, pos, font, color, max_width=None, line_spacing_factor=1.0):
    words = [word.split(' ') for word in text.splitlines()]
    space = font.size(' ')[0]
    x, y = pos
    line_height = font.get_linesize() * line_spacing_factor

    for line_words in words:
        current_line_text = ""
        current_line_width = 0
        line_start_x = x
        
        temp_line_parts = []
        for word in line_words:
            word_width = font.size(word)[0]
            if max_width and current_line_width + word_width >= max_width and current_line_width > 0:
                # Render current line part and start new
                line_surface = font.render(" ".join(temp_line_parts), True, color)
                surface.blit(line_surface, (line_start_x, y))
                y += line_height
                temp_line_parts = [word]
                current_line_width = word_width + space
            else:
                temp_line_parts.append(word)
                current_line_width += word_width + space
        
        # Render the last part of the line
        if temp_line_parts:
            line_surface = font.render(" ".join(temp_line_parts), True, color)
            surface.blit(line_surface, (line_start_x, y))
        y += line_height


# --- Game Initialization ---
game_state = HarmoniesGameState()
human_player_id = 0 # Human is Player 0
ai_player_id = 1    # AI is Player 1

# --- Calculate Screen Dimensions ---
single_board_width_approx, single_board_height_approx = get_board_dimensions(VALID_HEXES)
single_board_width_approx += int(HEX_WIDTH * 0.5)
single_board_height_approx += int(HEX_HEIGHT * 0.5)

SCREEN_WIDTH = (SIDE_PADDING * 2) + (single_board_width_approx * 2) + BOARD_SPACING
SCREEN_HEIGHT = TOP_PADDING + single_board_height_approx + INFO_PANEL_HEIGHT + SIDE_PADDING

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Harmonies GUI")

# --- Calculate Board Origins ---
board0_visual_center_x = SIDE_PADDING + single_board_width_approx // 2
board0_visual_center_y = TOP_PADDING + single_board_height_approx // 2

board1_visual_center_x = board0_visual_center_x + single_board_width_approx + BOARD_SPACING
board1_visual_center_y = board0_visual_center_y

# --- Initialize Text Input Box ---
input_box_y_offset = 65 # How far down from the top of the info panel
input_box_height = 30
input_box_width = SCREEN_WIDTH // 3
text_input_box = TextInputBox(
    SCREEN_WIDTH // 2 - input_box_width // 2,
    SCREEN_HEIGHT - INFO_PANEL_HEIGHT + input_box_y_offset, # Position within info panel
    input_box_width, input_box_height, game_font_medium, prompt="Choice:" # Using medium font
)
input_active_for_current_turn = False

# --- Main Game Loop ---
running = True
action_prompt = "" # For messages in the info panel

while running:
    submitted_value_from_box = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        
        if input_active_for_current_turn: # Only process for text box if it should be active
            submitted_value_from_box = text_input_box.handle_event(event)

    # --- Game Logic & Human Input ---
    current_player_is_human = (game_state.get_current_player() == human_player_id)
    game_is_not_over = not game_state.is_game_over()

    if current_player_is_human and game_is_not_over:
        if not input_active_for_current_turn: # Time to activate input for this turn
            legal_moves = game_state.get_legal_moves()
            if not legal_moves:
                action_prompt = "No legal moves for you!"
                # Here you might want to automatically advance the game if human has no moves
                # e.g., game_state._end_turn_actions() or similar logic
            else:
                action_prompt = f"Your turn ({game_state.turn_phase}). Enter choice # in box."
                text_input_box.is_active = True
                text_input_box.user_text = ""
                input_active_for_current_turn = True

                # Print options to CONSOLE for reference
                print(f"\n--- YOUR TURN (Phase: {game_state.turn_phase}) ---")
                print("--- Enter the number for your choice into the Pygame text box ---")
                if game_state.turn_phase == "choose_pile":
                    for i, pile_idx in enumerate(legal_moves):
                        if 0 <= pile_idx < len(game_state.available_piles):
                            print(f"  {i+1}: Pile {pile_idx} ({'/'.join(game_state.available_piles[pile_idx])})")
                elif game_state.turn_phase.startswith("place_tile"):
                    print(f"Your hand: {game_state.tiles_in_hand}")
                    for i, (tile_type, coord) in enumerate(legal_moves):
                        print(f"  {i+1}: Place {tile_type.upper()} at {coord}")
        
        if submitted_value_from_box is not None: # Enter was pressed in the text box
            legal_moves = game_state.get_legal_moves() # Get fresh list
            chosen_action = None
            try:
                choice_num = int(submitted_value_from_box) - 1
                if legal_moves and 0 <= choice_num < len(legal_moves):
                    chosen_action = legal_moves[choice_num]
                    game_state = game_state.apply_move(chosen_action)
                    action_prompt = "Move applied."
                else:
                    action_prompt = "Invalid choice number. Try again."
            except ValueError:
                action_prompt = "Invalid input (not a number). Try again."
            except Exception as e:
                print(f"Error applying move {chosen_action if chosen_action else 'N/A'}: {e}")
                action_prompt = f"Error: {e}"
            
            text_input_box.user_text = "" # Clear text box after submission attempt
            # Keep input_active_for_current_turn = True if input was bad, so user can retry
            # Deactivate and reset only on successful move
            if chosen_action:
                text_input_box.is_active = False
                input_active_for_current_turn = False

    # --- Game Logic (AI Turn - Placeholder) ---
    elif game_state.get_current_player() == ai_player_id and game_is_not_over:
        text_input_box.is_active = False # Ensure human input box is not active
        input_active_for_current_turn = False
        action_prompt = "AI is thinking..." # This will be displayed
        
        # Placeholder: AI makes a random valid move after a short delay
        pygame.time.wait(500) # Simulate thinking
        legal_ai_moves = game_state.get_legal_moves()
        if legal_ai_moves:
            ai_chosen_action = random.choice(legal_ai_moves)
            print(f"AI chose (randomly): {ai_chosen_action}")
            try:
                game_state = game_state.apply_move(ai_chosen_action)
                action_prompt = "AI moved."
            except Exception as e:
                print(f"Error in AI move: {e}")
                action_prompt = "AI Error."
        else:
            action_prompt = "AI has no legal moves."
            # If AI has no moves, and it's a placement phase with an empty hand, advance its phase
            if game_state.turn_phase.startswith("place_tile") and not game_state.tiles_in_hand:
                if game_state.turn_phase == "place_tile_1": game_state.turn_phase = "place_tile_2"
                elif game_state.turn_phase == "place_tile_2": game_state.turn_phase = "place_tile_3"
                elif game_state.turn_phase == "place_tile_3": game_state._end_turn_actions()
            elif not game_state.get_legal_moves() and game_state.turn_phase == "choose_pile":
                # This implies game might be over if AI cannot choose a pile and piles are empty
                print("AI cannot choose a pile and has no moves. Checking game end.")
                game_state._end_turn_actions() # This will trigger game over checks


    # --- Update and Drawing ---
    text_input_box.update() # Update for cursor blinking

    screen.fill(COLOR_BACKGROUND)

    # Draw Boards
    title0_str = f"Player {human_player_id} (You)"
    if game_state.get_current_player() == human_player_id and not game_state.is_game_over(): title0_str += " - YOUR TURN"
    title0 = game_font_medium.render(title0_str, True, (0,0,0))
    screen.blit(title0, (board0_visual_center_x - title0.get_width()//2, board0_visual_center_y - single_board_height_approx//2 - 30))
    draw_hex_grid(screen, board0_visual_center_x, board0_visual_center_y, VALID_HEXES, game_state.player_boards[human_player_id], game_font_small)

    title1_str = f"Player {ai_player_id} (AI)"
    if game_state.get_current_player() == ai_player_id and not game_state.is_game_over(): title1_str += " - AI'S TURN"
    title1 = game_font_medium.render(title1_str, True, (0,0,0))
    screen.blit(title1, (board1_visual_center_x - title1.get_width()//2, board1_visual_center_y - single_board_height_approx//2 - 30))
    draw_hex_grid(screen, board1_visual_center_x, board1_visual_center_y, VALID_HEXES, game_state.player_boards[ai_player_id], game_font_small)

    # Draw Info Panel
    info_panel_rect = pygame.Rect(0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, (215, 205, 195), info_panel_rect) # Slightly different color
    pygame.draw.line(screen, (150,150,150), (info_panel_rect.left, info_panel_rect.top), (info_panel_rect.right, info_panel_rect.top), 2)

    info_y_current = info_panel_rect.y + 10
    line_spacing = game_font_small.get_linesize()
    col1_x = info_panel_rect.x + 20
    col2_x = info_panel_rect.centerx + 10

    # Game Status
    status_text = f"Player: {game_state.get_current_player()} | Phase: {game_state.turn_phase}"
    screen.blit(game_font_medium.render(status_text, True, (0,0,0)), (col1_x, info_y_current))
    info_y_current += line_spacing * 1.5

    # Action Prompt
    if action_prompt:
        prompt_color = (200,0,0) if "Error" in action_prompt or "Invalid" in action_prompt else (0,0,180)
        prompt_surf = game_font_small.render(action_prompt, True, prompt_color)
        screen.blit(prompt_surf, (col1_x, info_y_current))
    info_y_current += line_spacing * 1.5 # Space for prompt or just move to next item

    # Piles Info
    available_piles_str = "Available Piles:\n"
    if game_state.available_piles:
        for i, pile in enumerate(game_state.available_piles):
            available_piles_str += f"  {i}: {'/'.join(p[:1].upper()+p[1:] for p in pile)}\n" # Nicer formatting
    else: available_piles_str += "  (None)\n"
    draw_text_multiline(screen, available_piles_str, (col1_x, info_y_current), game_font_small, (0,0,0), max_width=SCREEN_WIDTH//2 - 40)
    
    # Hand and Bag Info (in second column)
    info_y_col2_current = info_panel_rect.y + 10
    if game_state.turn_phase.startswith("place_tile") and game_state.get_current_player() == human_player_id:
        hand_str = f"Your Hand: {', '.join(game_state.tiles_in_hand) if game_state.tiles_in_hand else '(Empty)'}"
        screen.blit(game_font_small.render(hand_str, True, (0,0,0)), (col2_x, info_y_col2_current))
        info_y_col2_current += line_spacing * 1.5

    bag_str = "Tile Bag:\n"
    non_empty_bag_items = {k:v for k,v in game_state.tile_bag.items() if v > 0}
    if non_empty_bag_items:
        for tile, count in sorted(non_empty_bag_items.items()):
            bag_str += f"  {tile.capitalize()}: {count}\n"
    else: bag_str += "  (Empty)\n"
    draw_text_multiline(screen, bag_str, (col2_x, info_y_col2_current), game_font_small, (0,0,0), max_width=SCREEN_WIDTH//2 - 40)

    # Draw Text Input Box
    text_input_box.draw(screen) # Always draw, its appearance changes if active

    # Game Over Message
    if game_state.is_game_over():
        text_input_box.is_active = False # Ensure not active
        input_active_for_current_turn = False
        winner_text = ""
        outcome = game_state.get_game_outcome()
        # ... (game over text as before) ...
        if outcome == 1: winner_text = f"Player 0 WINS! Scores: P0:{game_state.final_scores[0]} P1:{game_state.final_scores[1]}"
        elif outcome == -1: winner_text = f"Player 1 WINS! Scores: P0:{game_state.final_scores[0]} P1:{game_state.final_scores[1]}"
        else: winner_text = f"It's a DRAW! Scores: P0:{game_state.final_scores[0]} P1:{game_state.final_scores[1]}"
        winner_surf = game_font_large.render(winner_text, True, (200, 0, 0))
        screen.blit(winner_surf, (SCREEN_WIDTH // 2 - winner_surf.get_width() // 2, TOP_PADDING // 2 - 10 ))


    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
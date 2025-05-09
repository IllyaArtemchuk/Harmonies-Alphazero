import pygame
import sys
import os 
import random
import math 

# --- CHOOSE YOUR HEX ORIENTATION ---
# (Keep your chosen orientation import from hex_utils.py)
from hex_utils import (
    draw_hex_grid_pointy_topped as draw_hex_grid,
    get_board_dimensions_pointy_topped as get_board_dimensions,
    draw_hexagon_pointy_topped as draw_hexagon,
    COLOR_BACKGROUND, HEX_SIZE, HEX_HEIGHT, HEX_WIDTH, TILE_COLOR_MAP, COLOR_TEXT, COLOR_HEX_BORDER
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import VALID_HEXES, TILE_TYPES, NUM_PILES, PILE_SIZE # Added NUM_PILES, PILE_SIZE
from harmonies_engine import HarmoniesGameState

# --- Pygame Setup ---
pygame.init()
pygame.font.init()

# Screen dimensions & UI element sizes
INFO_PANEL_HEIGHT = 250 # Increased for hand/piles display
BOARD_SPACING = 70
SIDE_PADDING = 50
TOP_PADDING = 50

PILE_RECT_WIDTH = 100
PILE_RECT_HEIGHT = 60 # Height for 3 tiles + padding
PILE_SPACING = 20
TILE_IN_PILE_SIZE = 15 # Small rects for tiles within a pile display

SMALL_HEX_SIZE = 15 # Radius for pile/hand tiles
SMALL_HEX_WIDTH_POINTY = math.sqrt(3) * SMALL_HEX_SIZE # If using pointy for small hexes
SMALL_HEX_HEIGHT_POINTY = SMALL_HEX_SIZE * 2

HAND_TILE_RECT_SIZE = 40
HAND_TILE_SPACING = 10

# Game Fonts
game_font_tiny = pygame.font.SysFont("Arial", 12)
game_font_small = pygame.font.SysFont("Arial", 16)
game_font_medium = pygame.font.SysFont("Arial", 20)
game_font_large = pygame.font.SysFont("Arial", 24)

clock = pygame.time.Clock()

def pixel_to_axial_pointy(x, y, origin_x, origin_y, main_hex_size):
    # Adjust x, y relative to the grid origin
    x_rel = x - origin_x
    y_rel = y - origin_y

    # Convert pixel to fractional axial coordinates (for pointy-topped)
    q_frac = (math.sqrt(3)/3 * x_rel - 1./3 * y_rel) / main_hex_size
    r_frac = (2./3 * y_rel) / main_hex_size
    
    # Cube coordinates for rounding
    x_cube = q_frac
    z_cube = r_frac
    y_cube = -x_cube - z_cube
    
    rx = round(x_cube)
    ry = round(y_cube)
    rz = round(z_cube)
    
    x_diff = abs(rx - x_cube)
    y_diff = abs(ry - y_cube)
    z_diff = abs(rz - z_cube)
    
    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry
        
    return int(rx), int(rz) # q = x_cube_rounded, r = z_cube_rounded

# --- TextInputBox Class (Keep as is from previous version) ---
class TextInputBox:
    # ... (Full TextInputBox class code from previous response) ...
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
        submitted_text = None 
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_active = True
            else:
                self.is_active = False 
        
        if self.is_active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                submitted_text = self.user_text
            elif event.key == pygame.K_BACKSPACE:
                self.user_text = self.user_text[:-1]
            else:
                if event.unicode.isalnum() or event.unicode in ['.', '-', ' ']:
                    self.user_text += event.unicode
            self.text_surface = self.font.render(self.user_text, True, self.text_color)
        return submitted_text

    def update(self):
        if self.is_active:
            self.cursor_timer += clock.get_time() 
            if self.cursor_timer >= 500: 
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0
        else:
            self.cursor_visible = False

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

# --- Helper to draw multiline text (Keep as is) ---
def draw_text_multiline(surface, text, pos, font, color, max_width=None, line_spacing_factor=1.0):
    # ... (Full draw_text_multiline class code from previous response) ...
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
                line_surface = font.render(" ".join(temp_line_parts), True, color)
                surface.blit(line_surface, (line_start_x, y))
                y += line_height
                temp_line_parts = [word]
                current_line_width = word_width + space
            else:
                temp_line_parts.append(word)
                current_line_width += word_width + space
        
        if temp_line_parts:
            line_surface = font.render(" ".join(temp_line_parts), True, color)
            surface.blit(line_surface, (line_start_x, y))
        y += line_height


# --- Game Initialization ---
game_state = HarmoniesGameState()
human_player_id = 0
ai_player_id = 1

# --- Calculate Screen Dimensions ---
single_board_width_approx, single_board_height_approx = get_board_dimensions(VALID_HEXES)
single_board_width_approx += int(HEX_WIDTH * 0.5)
single_board_height_approx += int(HEX_HEIGHT * 0.5)

SCREEN_WIDTH = (SIDE_PADDING * 2) + (single_board_width_approx * 2) + BOARD_SPACING
SCREEN_HEIGHT = TOP_PADDING + single_board_height_approx + INFO_PANEL_HEIGHT + SIDE_PADDING

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Harmonies GUI")

# Board Origins
board0_visual_center_x = SIDE_PADDING + single_board_width_approx // 2
board0_visual_center_y = TOP_PADDING + single_board_height_approx // 2
board1_visual_center_x = board0_visual_center_x + single_board_width_approx + BOARD_SPACING
board1_visual_center_y = board0_visual_center_y

# Text Input Box (No longer used for primary input if we have clicking)
# text_input_box = TextInputBox(...) # We might remove or repurpose this later

# --- NEW: Pile and Hand Rects ---
pile_rects = [] # List to store pygame.Rect for each pile
hand_tile_rects = [] # List to store pygame.Rect for each tile in hand
selected_hand_tile_index = None # To track which tile in hand is selected for placement
selected_hand_tile_type = None  # The actual tile type string

# Calculate positions for piles in the info panel
total_piles_width = (NUM_PILES * PILE_RECT_WIDTH) + ((NUM_PILES - 1) * PILE_SPACING)
piles_start_x = (SCREEN_WIDTH - total_piles_width) // 2
piles_y = SCREEN_HEIGHT - INFO_PANEL_HEIGHT + 20 # Top of info panel + padding

for i in range(NUM_PILES):
    rect_x = piles_start_x + i * (PILE_RECT_WIDTH + PILE_SPACING)
    pile_rects.append(pygame.Rect(rect_x, piles_y, PILE_RECT_WIDTH, PILE_RECT_HEIGHT))

# --- Main Game Loop ---
running = True
action_prompt = "Welcome! Click a pile to choose."

while running:
    mouse_pos = pygame.mouse.get_pos()
    clicked_pile_idx = None
    # We will add clicked_hex_coord and clicked_hand_tile_idx later

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
            if game_state.get_current_player() == human_player_id and not game_state.is_game_over():
                # 1. Check for Pile Clicks
                if game_state.turn_phase == "choose_pile":
                    for i, pile_r in enumerate(pile_rects):
                        if i < len(game_state.available_piles): # Only check existing piles
                            if pile_r.collidepoint(mouse_pos):
                                clicked_pile_idx = i # This is the actual index in game_state.available_piles
                                break
                
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
            if game_state.get_current_player() == human_player_id and not game_state.is_game_over():
                # 1. Check for Pile Clicks
                if game_state.turn_phase == "choose_pile":
                    for i, pile_r_actual_clickable_area in enumerate(pile_rects): # pile_rects now stores the bounding box of the pile
                        if i < len(game_state.available_piles):
                            if pile_r_actual_clickable_area.collidepoint(mouse_pos):
                                clicked_pile_idx = i
                                break
                # 2. Check for Hand Tile Clicks
                elif game_state.turn_phase.startswith("place_tile"):
                    newly_selected_hand_idx = None
                    for i, hand_r_clickable_area in enumerate(hand_tile_rects): # hand_tile_rects are bounding boxes
                        if hand_r_clickable_area.collidepoint(mouse_pos):
                            newly_selected_hand_idx = i
                            break
                    if newly_selected_hand_idx is not None:
                        if selected_hand_tile_index == newly_selected_hand_idx: # Clicked same tile again
                            selected_hand_tile_index = None # Deselect
                            selected_hand_tile_type = None
                            action_prompt = "Tile deselected. Select a tile from hand."
                        else:
                            selected_hand_tile_index = newly_selected_hand_idx
                            selected_hand_tile_type = game_state.tiles_in_hand[selected_hand_tile_index]
                            action_prompt = f"{selected_hand_tile_type.upper()} selected. Click a hex to place."
                
                # 3. Check for Hex Grid Clicks (for placement, if a hand tile is selected)
                if game_state.turn_phase.startswith("place_tile") and selected_hand_tile_type is not None:
                    # Check clicks on Player 0's board (human_player_id's board)
                    # Convert mouse_pos to q,r for board0
                    # Note: HEX_SIZE here is for the *main board hexes*
                    q_clicked, r_clicked = pixel_to_axial_pointy(mouse_pos[0], mouse_pos[1],
                                                                  board0_visual_center_x, board0_visual_center_y,
                                                                  HEX_SIZE)
                    clicked_coord = (q_clicked, r_clicked)

                    if clicked_coord in VALID_HEXES: # Check if the derived axial coord is a valid hex
                        potential_action = (selected_hand_tile_type, clicked_coord)
                        current_legal_moves = game_state.get_legal_moves() # These are (tile_type, coord)

                        if potential_action in current_legal_moves:
                            try:
                                game_state = game_state.apply_move(potential_action)
                                action_prompt = f"Placed {selected_hand_tile_type.upper()} at {clicked_coord}."
                                selected_hand_tile_index = None # Reset selection
                                selected_hand_tile_type = None
                            except Exception as e:
                                print(f"Error applying placement {potential_action}: {e}")
                                action_prompt = f"Placement Error: {e}"
                        else:
                            action_prompt = f"Cannot place {selected_hand_tile_type.upper()} at {clicked_coord}. Illegal move."

    # --- Game Logic & Human Input Processing ---
    current_player_is_human = (game_state.get_current_player() == human_player_id)
    game_is_not_over = not game_state.is_game_over()

    if current_player_is_human and game_is_not_over:
        legal_moves = game_state.get_legal_moves()

        if game_state.turn_phase == "choose_pile":
            action_prompt = "Your Turn: Choose a Pile"
            if clicked_pile_idx is not None: # A pile was clicked
                if clicked_pile_idx in legal_moves: # legal_moves for pile phase are pile indices
                    try:
                        game_state = game_state.apply_move(clicked_pile_idx)
                        action_prompt = f"Pile {clicked_pile_idx} chosen. Hand: {game_state.tiles_in_hand}"
                        selected_hand_tile_index = None # Reset selection for placement
                    except Exception as e:
                        print(f"Error applying pile choice {clicked_pile_idx}: {e}")
                        action_prompt = f"Error: {e}"
                else:
                    action_prompt = "Not a valid pile choice."
        
        elif game_state.turn_phase.startswith("place_tile"):
            action_prompt = "Your Turn: Select a tile from hand, then a hex to place."
            if not game_state.tiles_in_hand and legal_moves: # Should have tiles if legal moves exist
                # This case needs careful thought - how can there be legal placement moves without tiles in hand?
                # It implies an issue in get_legal_moves or game state.
                # For now, if hand is empty but phase is placement, try to auto-advance
                print(f"Warning: Placement phase but hand is empty. Legal moves: {legal_moves}")
                if game_state.turn_phase == "place_tile_1": game_state.turn_phase = "place_tile_2"
                elif game_state.turn_phase == "place_tile_2": game_state.turn_phase = "place_tile_3"
                elif game_state.turn_phase == "place_tile_3": game_state._end_turn_actions()

            # TODO: Process clicked_hand_tile_idx to set selected_hand_tile_index
            # TODO: Process clicked_hex_coord with selected_hand_tile_index to make a placement move
            pass # Placeholder for placement logic

        elif not legal_moves:
            action_prompt = "No legal moves for you!"
            # Auto-advance if stuck in placement with empty hand
            if game_state.turn_phase.startswith("place_tile") and not game_state.tiles_in_hand:
                if game_state.turn_phase == "place_tile_1": game_state.turn_phase = "place_tile_2"
                elif game_state.turn_phase == "place_tile_2": game_state.turn_phase = "place_tile_3"
                elif game_state.turn_phase == "place_tile_3": game_state._end_turn_actions()


    # --- Game Logic (AI Turn - Placeholder) ---
    elif game_state.get_current_player() == ai_player_id and game_is_not_over:
        action_prompt = "AI is thinking..."
        pygame.time.wait(300) # Simulate thinking
        legal_ai_moves = game_state.get_legal_moves()
        if legal_ai_moves:
            ai_chosen_action = random.choice(legal_ai_moves) # AI picks randomly for now
            print(f"AI chose (randomly): {ai_chosen_action}")
            try:
                game_state = game_state.apply_move(ai_chosen_action)
                action_prompt = "AI moved."
            except Exception as e:
                print(f"Error in AI move: {e}")
                action_prompt = "AI Error."
        else:
            action_prompt = "AI has no legal moves."
            if game_state.turn_phase.startswith("place_tile") and not game_state.tiles_in_hand:
                if game_state.turn_phase == "place_tile_1": game_state.turn_phase = "place_tile_2"
                elif game_state.turn_phase == "place_tile_2": game_state.turn_phase = "place_tile_3"
                elif game_state.turn_phase == "place_tile_3": game_state._end_turn_actions()
            elif not game_state.get_legal_moves(): game_state._end_turn_actions()


    # --- Drawing ---
    screen.fill(COLOR_BACKGROUND)

    # Draw Boards (same as before)
    title0_str = f"Player {human_player_id} (You)"
    if game_state.get_current_player() == human_player_id and not game_state.is_game_over(): title0_str += " - YOUR TURN"
    title0 = game_font_medium.render(title0_str, True, (0,0,0))
    screen.blit(title0, (board0_visual_center_x - title0.get_width()//2, board0_visual_center_y - single_board_height_approx//2 - 30))
    draw_hex_grid(screen, board0_visual_center_x, board0_visual_center_y, VALID_HEXES, game_state.player_boards[human_player_id], game_font_tiny) # Smaller font for coords/stacks

    title1_str = f"Player {ai_player_id} (AI)"
    if game_state.get_current_player() == ai_player_id and not game_state.is_game_over(): title1_str += " - AI'S TURN"
    title1 = game_font_medium.render(title1_str, True, (0,0,0))
    screen.blit(title1, (board1_visual_center_x - title1.get_width()//2, board1_visual_center_y - single_board_height_approx//2 - 30))
    draw_hex_grid(screen, board1_visual_center_x, board1_visual_center_y, VALID_HEXES, game_state.player_boards[ai_player_id], game_font_tiny)


    # --- Info Panel ---
    info_panel_rect = pygame.Rect(0, SCREEN_HEIGHT - INFO_PANEL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, (215, 205, 195), info_panel_rect)
    pygame.draw.line(screen, (150,150,150), (info_panel_rect.left, info_panel_rect.top), (info_panel_rect.right, info_panel_rect.top), 2)

    info_y_current = info_panel_rect.y + 10
    line_spacing_small = game_font_small.get_linesize()
    line_spacing_medium = game_font_medium.get_linesize()
    col1_x = info_panel_rect.x + 20
    col2_x = info_panel_rect.centerx + 10
    
    # Game Status
    status_text = f"Player: {game_state.get_current_player()} | Phase: {game_state.turn_phase}"
    screen.blit(game_font_medium.render(status_text, True, (0,0,0)), (col1_x, info_y_current))
    info_y_current += line_spacing_medium * 1.2

    # Action Prompt
    if action_prompt:
        prompt_color = (200,0,0) if "Error" in action_prompt or "Invalid" in action_prompt else (0,0,100)
        prompt_surf = game_font_small.render(action_prompt, True, prompt_color)
        screen.blit(prompt_surf, (col1_x, info_y_current))
    info_y_current += line_spacing_small * 2.5 # INCREASED Y OFFSET HERE


    # --- NEW: Draw Piles ---
    piles_title_surf = game_font_small.render("Available Piles:", True, COLOR_TEXT)
    screen.blit(piles_title_surf, (piles_start_x, piles_y - line_spacing_small - 5))
    
    pile_display_box_width = max(SMALL_HEX_WIDTH_POINTY * 1.5, 60) # Width for one hex, or a min width
    pile_display_box_height = SMALL_HEX_HEIGHT_POINTY * PILE_SIZE + (PILE_SIZE-1)*2 # Height for stacked hexes

    total_piles_display_width = (NUM_PILES * pile_display_box_width) + ((NUM_PILES - 1) * PILE_SPACING)
    piles_start_x = (SCREEN_WIDTH - total_piles_display_width) // 2
    piles_y_draw = info_y_current # Use the updated info_y_current

    screen.blit(piles_title_surf, (piles_start_x, piles_y_draw - line_spacing_small - 5))
    piles_y_draw += line_spacing_small # Add space after title

    pile_rects.clear() # Rebuild clickable rects each frame

    for i in range(NUM_PILES):
        pile_box_x = piles_start_x + i * (pile_display_box_width + PILE_SPACING)
        pile_bounding_rect = pygame.Rect(pile_box_x, piles_y_draw, pile_display_box_width, pile_display_box_height)
        pile_rects.append(pile_bounding_rect) # Store the bounding box for clicking

        is_hovered = pile_bounding_rect.collidepoint(mouse_pos) and \
                     game_state.turn_phase == "choose_pile" and \
                     current_player_is_human

        border_color_pile = (255,165,0) if is_hovered else COLOR_HEX_BORDER
        pygame.draw.rect(screen, (230,230,220) if i < len(game_state.available_piles) else (180,180,180), pile_bounding_rect)
        pygame.draw.rect(screen, border_color_pile, pile_bounding_rect, 2 if is_hovered else 1)

        if i < len(game_state.available_piles):
            pile_data = game_state.available_piles[i]
            for tile_idx, tile_type_str in enumerate(pile_data):
                tile_color = TILE_COLOR_MAP.get(tile_type_str.lower(), (0,0,0))
                # Draw small hexes stacked vertically within the pile_bounding_rect
                # Assuming pointy-topped for these small hexes
                hex_center_x = pile_bounding_rect.centerx
                hex_center_y = pile_bounding_rect.y + SMALL_HEX_SIZE + tile_idx * (SMALL_HEX_HEIGHT_POINTY * 0.85) # 0.85 for tighter stacking

                # Use the aliased draw_hexagon from hex_utils which now takes size
                draw_hexagon(screen, tile_color, hex_center_x, hex_center_y, SMALL_HEX_SIZE, COLOR_HEX_BORDER,1)
        else:
            screen.blit(game_font_tiny.render("(empty)", True, COLOR_TEXT), (pile_bounding_rect.centerx-15, pile_bounding_rect.centery-5))
    
    info_y_current = piles_y_draw + pile_display_box_height + 15 # Update info_y_current


    # --- Draw Player's Hand (with hexes) ---
    hand_tile_rects.clear() # Rebuild clickable rects for hand tiles
    if game_state.get_current_player() == human_player_id and game_state.tiles_in_hand:
        hand_title_surf = game_font_small.render("Your Hand (Click to Select, Click Hex to Place):", True, COLOR_TEXT)
        
        # Calculate starting position for hand tiles to center them
        num_hand_tiles = len(game_state.tiles_in_hand)
        total_hand_width = (num_hand_tiles * HAND_HEX_WIDTH_POINTY) + \
                           (max(0, num_hand_tiles - 1) * HAND_TILE_SPACING)
        hand_start_x = (SCREEN_WIDTH - total_hand_width) // 2
        hand_y_draw = info_y_current # Use updated y

        screen.blit(hand_title_surf, (hand_start_x, hand_y_draw - line_spacing_small - 5))
        hand_y_draw += line_spacing_small

        for i, tile_type_str in enumerate(game_state.tiles_in_hand):
            # Assuming pointy-topped for hand tiles
            hex_center_x = hand_start_x + HAND_HEX_WIDTH_POINTY // 2 + i * (HAND_HEX_WIDTH_POINTY + HAND_TILE_SPACING)
            hex_center_y = hand_y_draw + HAND_HEX_HEIGHT_POINTY // 2
            
            # Create a clickable bounding box around the hand hex
            # For pointy, width is HEX_WIDTH_POINTY, height is HEX_HEIGHT_POINTY
            clickable_rect = pygame.Rect(
                hex_center_x - HAND_HEX_WIDTH_POINTY // 2,
                hex_center_y - HAND_HEX_HEIGHT_POINTY // 2,
                HAND_HEX_WIDTH_POINTY,
                HAND_HEX_HEIGHT_POINTY
            )
            hand_tile_rects.append(clickable_rect)

            tile_color = TILE_COLOR_MAP.get(tile_type_str.lower(), (0,0,0))
            is_selected = (selected_hand_tile_index == i)
            is_hovered = clickable_rect.collidepoint(mouse_pos) and game_state.turn_phase.startswith("place_tile")

            border_c = (0,255,0) if is_selected else ((255,165,0) if is_hovered else COLOR_HEX_BORDER)
            border_w = 3 if is_selected else (2 if is_hovered else 1)

            draw_hexagon(screen, tile_color, hex_center_x, hex_center_y, HAND_TILE_HEX_SIZE, border_c, border_w)
            # Optional: text on hand tile
            # hand_tile_text = game_font_tiny.render(tile_type_str[:1].upper(), True, COLOR_TEXT if sum(tile_color) > 384 else (255,255,255))
            # screen.blit(hand_tile_text, (tile_r.centerx - hand_tile_text.get_width()//2, tile_r.centery - hand_tile_text.get_height()//2))


    # Bag Info (can be drawn elsewhere or more compactly)
    # ... (keep bag drawing if you like, or integrate it more neatly) ...

    # Game Over Message
    if game_state.is_game_over():
        # ... (Game over text as before) ...
        winner_text = ""
        outcome = game_state.get_game_outcome()
        if outcome == 1: winner_text = f"Player 0 WINS! Scores: P0:{game_state.final_scores[0]} P1:{game_state.final_scores[1]}"
        elif outcome == -1: winner_text = f"Player 1 WINS! Scores: P0:{game_state.final_scores[0]} P1:{game_state.final_scores[1]}"
        else: winner_text = f"It's a DRAW! Scores: P0:{game_state.final_scores[0]} P1:{game_state.final_scores[1]}"
        winner_surf = game_font_large.render(winner_text, True, (200, 0, 0))
        screen.blit(winner_surf, (SCREEN_WIDTH // 2 - winner_surf.get_width() // 2, TOP_PADDING // 2 - 10 ))


    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
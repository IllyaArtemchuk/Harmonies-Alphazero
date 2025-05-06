# GUI/hex_utils.py
import pygame
import math

# --- Configuration ---
HEX_SIZE = 30  # Radius of the hexagon
HEX_WIDTH = HEX_SIZE * 2
HEX_HEIGHT = math.sqrt(3) * HEX_SIZE

# Colors
COLOR_BACKGROUND = (240, 230, 220) # Light parchment
COLOR_HEX_EMPTY = (200, 200, 200)  # Light grey
COLOR_HEX_BORDER = (50, 50, 50)    # Dark grey
COLOR_TEXT = (10, 10, 10)

# Tile Colors (simple representation for now)
TILE_COLOR_MAP = {
    "water": (100, 150, 255),
    "plant": (50, 200, 50),
    "wood": (139, 69, 19),
    "stone": (128, 128, 128),
    "building": (255, 80, 80),
    "field": (240, 230, 140),
    "empty": COLOR_HEX_EMPTY # For placeholder
}

# --- Axial to Pixel Conversion ---
def axial_to_pixel(q, r, origin_x, origin_y):
    """Converts axial hex coordinates to pixel coordinates."""
    x = origin_x + HEX_SIZE * (3./2. * q)
    y = origin_y + HEX_SIZE * (math.sqrt(3)/2. * q + math.sqrt(3) * r)
    return int(x), int(y)

# --- Drawing Functions ---
def draw_hexagon(surface, color, x_center, y_center, border_color=None, border_width=2):
    """Draws a single hexagon."""
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30 # -30 to make flat top
        angle_rad = math.pi / 180 * angle_deg
        points.append((x_center + HEX_SIZE * math.cos(angle_rad),
                       y_center + HEX_SIZE * math.sin(angle_rad)))
    pygame.draw.polygon(surface, color, points)
    if border_color:
        pygame.draw.polygon(surface, border_color, points, border_width)

def draw_hex_grid(surface, origin_x, origin_y, valid_hexes_coords, player_board_data=None, font=None):
    """
    Draws a full hex grid.
    valid_hexes_coords: A set or list of (q, r) tuples defining the board shape.
    player_board_data: A dictionary {(q,r): [tile_stack]} for the player's tiles.
    """
    if player_board_data is None:
        player_board_data = {}

    for q, r in valid_hexes_coords:
        x_center, y_center = axial_to_pixel(q, r, origin_x, origin_y)

        # Determine tile stack and top tile
        tile_stack = player_board_data.get((q, r), [])
        top_tile_type = tile_stack[-1] if tile_stack else "empty"
        hex_color = TILE_COLOR_MAP.get(top_tile_type.lower(), COLOR_HEX_EMPTY) # Use .lower() for safety

        draw_hexagon(surface, hex_color, x_center, y_center, COLOR_HEX_BORDER)

        # Optional: Draw stack height or coordinates
        if font:
            # Stack height
            if len(tile_stack) > 1:
                stack_text = font.render(str(len(tile_stack)), True, COLOR_TEXT)
                surface.blit(stack_text, (x_center - stack_text.get_width() // 2,
                                         y_center - stack_text.get_height() // 2 - 10)) # Adjust position
            # Coordinates (for debugging)
            # coord_text = font.render(f"{q},{r}", True, COLOR_TEXT)
            # surface.blit(coord_text, (x_center - coord_text.get_width() // 2,
            #                           y_center - coord_text.get_height() // 2))

def get_board_dimensions(valid_hexes_coords):
    """Calculates the approximate pixel dimensions needed for a hex grid."""
    if not valid_hexes_coords:
        return 0, 0

    min_q = min(q for q, r in valid_hexes_coords)
    max_q = max(q for q, r in valid_hexes_coords)
    min_r = min(r for q, r in valid_hexes_coords)
    max_r = max(r for q, r in valid_hexes_coords)

    # Get pixel coordinates for corners (approximate)
    # Top-left-most possible hex (e.g., high negative q, low r)
    # Bottom-right-most possible hex (e.g., high positive q, high r)
    # This is a rough estimate for padding purposes
    dummy_origin_x, dummy_origin_y = HEX_WIDTH, HEX_HEIGHT # Some offset

    p_min_q_min_r = axial_to_pixel(min_q, min_r, dummy_origin_x, dummy_origin_y)
    p_max_q_max_r = axial_to_pixel(max_q, max_r, dummy_origin_x, dummy_origin_y)

    # Approximate width and height, adding padding for hex extent
    width = abs(p_max_q_max_r[0] - p_min_q_min_r[0]) + HEX_WIDTH * 2
    height = abs(p_max_q_max_r[1] - p_min_q_min_r[1]) + HEX_HEIGHT * 2
    return int(width), int(height)
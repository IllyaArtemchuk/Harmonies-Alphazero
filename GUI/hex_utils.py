# GUI/hex_utils.py
import pygame
import math

# --- Configuration ---
HEX_SIZE = 30  # Radius of the hexagon (distance from center to a vertex)

# For pointy-topped hexagons:
HEX_WIDTH = math.sqrt(3) * HEX_SIZE # Full width of the hexagon
HEX_HEIGHT = HEX_SIZE * 2          # Full height of the hexagon

# Colors (same as before)
COLOR_BACKGROUND = (240, 230, 220)
COLOR_HEX_EMPTY = (200, 200, 200)
COLOR_HEX_BORDER = (50, 50, 50)
COLOR_TEXT = (10, 10, 10)
TILE_COLOR_MAP = {
    "water": (100, 150, 255), "plant": (50, 200, 50), "wood": (139, 69, 19),
    "stone": (128, 128, 128), "building": (255, 80, 80), "field": (240, 230, 140),
    "empty": COLOR_HEX_EMPTY
}

# --- Axial to Pixel Conversion for POINTY-TOPPED Hexagons ---
def axial_to_pixel_pointy_topped(q, r, origin_x, origin_y):
    """Converts axial hex coordinates to pixel coordinates for a pointy-topped grid."""
    x = origin_x + HEX_SIZE * (math.sqrt(3) * q + math.sqrt(3)/2. * r)
    y = origin_y + HEX_SIZE * (3./2. * r)
    return int(x), int(y)

# --- Drawing Functions ---
def draw_hexagon_pointy_topped(surface, color, x_center, y_center, border_color=None, border_width=2):
    """Draws a single pointy-topped hexagon."""
    points = []
    for i in range(6):
        # For pointy-topped, angles start offset by 30 degrees (e.g., 30, 90, 150, ...)
        # Or more simply, 0, 60, 120... and then rotate the whole coordinate system effectively.
        # Let's use the common angle offset method:
        angle_deg = 60 * i + 30 # +30 to make pointy top
        angle_rad = math.pi / 180 * angle_deg
        points.append((x_center + HEX_SIZE * math.cos(angle_rad),
                       y_center + HEX_SIZE * math.sin(angle_rad)))
    pygame.draw.polygon(surface, color, points)
    if border_color:
        pygame.draw.polygon(surface, border_color, points, border_width)

def draw_hex_grid_pointy_topped(surface, origin_x, origin_y, valid_hexes_coords, player_board_data=None, font=None):
    """
    Draws a full pointy-topped hex grid.
    valid_hexes_coords: A set or list of (q, r) tuples defining the board shape.
    player_board_data: A dictionary {(q,r): [tile_stack]} for the player's tiles.
    """
    if player_board_data is None:
        player_board_data = {}

    for q, r in valid_hexes_coords:
        # Use the pointy-topped conversion
        x_center, y_center = axial_to_pixel_pointy_topped(q, r, origin_x, origin_y)

        tile_stack = player_board_data.get((q, r), [])
        top_tile_type = tile_stack[-1] if tile_stack else "empty"
        hex_color = TILE_COLOR_MAP.get(top_tile_type.lower(), COLOR_HEX_EMPTY)

        # Use the pointy-topped drawing function
        draw_hexagon_pointy_topped(surface, hex_color, x_center, y_center, COLOR_HEX_BORDER)

        if font:
            if len(tile_stack) > 1:
                stack_text = font.render(str(len(tile_stack)), True, COLOR_TEXT)
                surface.blit(stack_text, (x_center - stack_text.get_width() // 2,
                                         y_center - stack_text.get_height() // 2 - 10))
            # coord_text = font.render(f"{q},{r}", True, COLOR_TEXT) # For debugging
            # surface.blit(coord_text, (x_center - coord_text.get_width() // 2,
            #                           y_center - coord_text.get_height() // 2))


def get_board_dimensions_pointy_topped(valid_hexes_coords):
    """Calculates approximate pixel dimensions for a pointy-topped hex grid."""
    if not valid_hexes_coords:
        return 0, 0

    pixel_coords = [axial_to_pixel_pointy_topped(q, r, 0, 0) for q,r in valid_hexes_coords]
    if not pixel_coords: return 0,0

    # For pointy-topped, HEX_WIDTH is sqrt(3)*size, HEX_HEIGHT is 2*size
    min_px_x = min(p[0] for p in pixel_coords) - (HEX_WIDTH / 2)
    max_px_x = max(p[0] for p in pixel_coords) + (HEX_WIDTH / 2)
    min_px_y = min(p[1] for p in pixel_coords) - HEX_SIZE # Account for hex radius (half height)
    max_px_y = max(p[1] for p in pixel_coords) + HEX_SIZE

    width = max_px_x - min_px_x + HEX_WIDTH * 0.5
    height = max_px_y - min_px_y + HEX_HEIGHT * 0.5
    
    return int(width), int(height)
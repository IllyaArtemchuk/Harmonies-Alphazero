TILE_TYPES = ["water", "plant", "wood", "stone", "building", "field"]

# Pre-defined hex grid with 5-4-5-4-5 pattern (23 hexes total)
VALID_HEXES = {
    # r = -2 (Top row, 5 hexes)
    (-1, -2),
    (0, -2),
    (1, -2),
    (2, -2),
    (3, -2),
    # r = -1 (Second row, 4 hexes)
    (-1, -1),
    (0, -1),
    (1, -1),
    (2, -1),
    # r = 0 (Middle row, 5 hexes)
    (-2, 0),
    (-1, 0),
    (0, 0),
    (1, 0),
    (2, 0),
    # r = 1 (Fourth row, 4 hexes)
    (-2, 1),
    (-1, 1),
    (0, 1),
    (1, 1),
    # r = 2 (Bottom row, 5 hexes)
    (-2, 2),
    (-1, 2),
    (0, 2),
    (1, 2),
    (2, 2),
}

AXIAL_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


WATER, PLANT, WOOD, STONE, BUILDING, FIELD = TILE_TYPES

INITIAL_BAG = {WATER: 23, PLANT: 19, WOOD: 21, STONE: 23, FIELD: 19, BUILDING: 15}
NUM_PILES = 5
PILE_SIZE = 3
NUM_HEXES = 23
EMPTY_HEX_END_THRESHOLD = 2

sorted_coords = sorted(list(VALID_HEXES))

coordinate_to_index_map = {coord: index for index, coord in enumerate(sorted_coords)}

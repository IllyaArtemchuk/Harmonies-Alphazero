from constants import *
from config import *
import numpy as np
import torch

def create_state_tensors(game_state):
    return (create_board_tensor(game_state), create_global_features(game_state))

def create_board_tensor(game_state):
    """
    Creates a spatial tensor representing board state, player, and phase.
    Output shape: (C, H, W) = (38, 5, 6)
    """
    # Define grid boundaries (assuming 5-4-5-4-5 grid)
    q_min, q_max = -2, 3 
    r_min, r_max = -2, 2
    
    # Calculate dimensions
    width = q_max - q_min + 1  # 6
    height = r_max - r_min + 1  # 5
    
    # Base channels: Board tiles (36) + Player (1) + Phase (1) = 38
    num_channels = (len(TILE_TYPES) * 3 * 2) + 1 + 1
    
    tensor = torch.zeros(num_channels, height, width, dtype=torch.float)
    
    # --- Create mask for valid hex positions ---
    valid_positions_mask = torch.zeros(height, width, dtype=torch.float)
    for q, r in VALID_HEXES:
        x = q - q_min
        y = r - r_min
        if 0 <= x < width and 0 <= y < height:
            valid_positions_mask[y, x] = 1.0 # Use 1.0 for float tensor

    # --- Fill in tile information (Channels 0-35) ---
    tile_channel_offset = len(TILE_TYPES) * 3 # 18 channels per player
    for player in [0, 1]:
        board = game_state.player_boards[player]
        player_offset = player * tile_channel_offset
        
        for (q, r), stack in board.items():
            x = q - q_min
            y = r - r_min
            if not (0 <= x < width and 0 <= y < height): continue
                
            for stack_pos, tile_type in enumerate(stack):
                if stack_pos >= 3: break # Max stack height encoding = 3
                
                try:
                    tile_idx = TILE_TYPES.index(tile_type)
                except ValueError:
                    print(f"Warning: Unknown tile type '{tile_type}' encountered in state.")
                    continue # Skip unknown tile types

                # Calculate channel index specific to tile type, stack position, and player
                channel_idx = player_offset + (tile_idx * 3) + stack_pos
                tensor[channel_idx, y, x] = 1.0

    # --- Add current player channel (Channel 36) ---
    player_channel_idx = tile_channel_offset * 2 # 36
    tensor[player_channel_idx, :, :] = float(game_state.current_player)

    # --- Add turn phase channel (Channel 37) ---
    phase_channel_idx = player_channel_idx + 1 # 37
    phase_list = ["choose_pile", "place_tile_1", "place_tile_2", "place_tile_3"]
    try:
        # Normalize phase index (0 to 3) -> (0.0 to 1.0)
        phase_val = phase_list.index(game_state.turn_phase) / 3.0 
    except ValueError:
        phase_val = 0.0 # Default for other phases like "game_over"
    tensor[phase_channel_idx, :, :] = phase_val

    # --- Apply valid hex mask to ALL channels ---
    # Broadcasting should work: (C, H, W) * (H, W) -> (C, H, W)
    tensor *= valid_positions_mask

    return tensor

def create_global_features(game_state):
    """
    Creates a single 1D tensor containing normalized global features:
    Available Piles, Tiles in Hand, Bag Counts.
    """
    
    # --- Available Piles Features (Size: NUM_PILES * len(TILE_TYPES) = 5 * 6 = 30) ---
    # Represents counts of each tile type in each available pile, normalized by max possible (3)
    pile_features = torch.zeros(NUM_PILES * len(TILE_TYPES), dtype=torch.float)
    for i in range(NUM_PILES):
        if i < len(game_state.available_piles): # Check if pile exists
            pile = game_state.available_piles[i]
            for tile_idx, tile_type in enumerate(TILE_TYPES):
                count = pile.count(tile_type)
                feature_idx = i * len(TILE_TYPES) + tile_idx
                pile_features[feature_idx] = count / PILE_SIZE # Normalize by max possible count (3)
        # else: leave features as 0 for non-existent piles

    # --- Tiles in Hand Features (Size: len(TILE_TYPES) = 6) ---
    # Represents counts of each tile type in hand, normalized by max possible (3)
    hand_features = torch.zeros(len(TILE_TYPES), dtype=torch.float)
    if game_state.tiles_in_hand: # Check if hand is not empty
       for tile_idx, tile_type in enumerate(TILE_TYPES):
           count = game_state.tiles_in_hand.count(tile_type)
           hand_features[tile_idx] = count / PILE_SIZE # Normalize by max possible hand size (3)
           
    # --- Bag Counts Features (Size: len(TILE_TYPES) = 6) ---
    # Represents remaining count of each tile type, normalized by initial count
    bag_features = torch.zeros(len(TILE_TYPES), dtype=torch.float)
    for tile_idx, tile_type in enumerate(TILE_TYPES):
        initial_count = INITIAL_BAG.get(tile_type, 1) # Avoid division by zero if somehow missing
        if initial_count > 0:
             bag_features[tile_idx] = game_state.tile_bag.get(tile_type, 0) / initial_count
        # else: leave as 0 if initial count was 0

    # --- Concatenate all global features ---
    global_features = torch.cat((pile_features, hand_features, bag_features), dim=0)

    # Total size = 30 (piles) + 6 (hand) + 6 (bag) = 42
    return global_features


def get_action_index(action):
    """ Maps a game action (pile_idx or (tile_idx, coord)) to a flat index (0-73). """
    # Example logic:
    if isinstance(action, int): # Pile choice
        return action # Assumes pile indices 0-4 match action indices 0-4
    elif isinstance(action, tuple) and len(action) == 2: # Placement (tile_idx, coord)
        tile_idx, coord = action
        # Need a consistent mapping from coord (q,r) to a linear index 0-22
        coord_map = coordinate_to_index_map # Precompute this mapping
        coord_idx = coord_map[coord]
        # Calculate flat index: 5 (piles) + tile_idx * 23 + coord_idx
        return 5 + (tile_idx * NUM_HEXES) + coord_idx
    else:
        raise ValueError(f"Invalid action format: {action}")
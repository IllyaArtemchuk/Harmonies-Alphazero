from constants import *
from config import *
import torch

def create_state_tensor(game_state):
    # Define grid boundaries based on your hex coordinates
    q_min, q_max = -2, 3
    r_min, r_max = -2, 2
    
    # Calculate dimensions
    width = q_max - q_min + 1  # 6
    height = r_max - r_min + 1  # 5
    
    # Number of channels needed for tile information
    tile_channels = len(TILE_TYPES) * 3 * 2  # 6 tile types × 3 stack positions × 2 players
    
    tensor = torch.zeros(tile_channels, height, width)
    
    # Process both players' boards
    for player in [0, 1]:
        board = game_state.player_boards[player]
        
        # Fill in tile information
        for (q, r), stack in board.items():
            # Convert hex coordinates to tensor indices
            x = q - q_min
            y = r - r_min
            
            # Skip if outside our grid
            if not (0 <= x < width and 0 <= y < height):
                continue
                
            # Encode each tile in the stack
            for stack_pos, tile_type in enumerate(stack):
                if stack_pos >= 3:  # Only handle stacks up to height 3
                    break
                
                # Calculate channel index
                tile_idx = TILE_TYPES.index(tile_type)
                channel_idx = player * (len(TILE_TYPES) * 3) + (tile_idx * 3) + stack_pos
                
                # Set the feature to 1
                tensor[channel_idx, y, x] = 1
    
    # Create mask for valid hex positions
    valid_positions_mask = torch.zeros(height, width)
    for q, r in VALID_HEXES:
        x = q - q_min
        y = r - r_min
        if 0 <= x < width and 0 <= y < height:
            valid_positions_mask[y, x] = 1
    
    # Apply mask to all channels
    for c in range(tensor.shape[0]):
        tensor[c] *= valid_positions_mask
    
    # Add current player channel
    player_channel = torch.full((1, height, width), game_state.current_player, dtype=torch.float)
    
    # Add turn phase encoding
    phase_list = ["choose_pile", "place_tile_1", "place_tile_2", "place_tile_3"]
    if game_state.turn_phase in phase_list:
        phase_idx = phase_list.index(game_state.turn_phase)
    else:
        phase_idx = 0  # Default for other phases like "game_over"
    phase_channel = torch.full((1, height, width), phase_idx / 3, dtype=torch.float)
    
    # Apply mask to additional channels
    player_channel *= valid_positions_mask
    phase_channel *= valid_positions_mask
    
    # Add channels for available piles (each pile as a separate channel)
    pile_channels = []
    for pile_idx, pile in enumerate(game_state.available_piles):
        for tile_type in TILE_TYPES:
            # Count occurrences of this tile type in the pile
            count = pile.count(tile_type)
            # Create a channel with the count value
            pile_channel = torch.full((1, height, width), count / 3, dtype=torch.float)
            pile_channel *= valid_positions_mask
            pile_channels.append(pile_channel)
    
    # Add channels for tiles in hand
    hand_channels = []
    for tile_type in TILE_TYPES:
        # Count occurrences of this tile type in the hand
        count = game_state.tiles_in_hand.count(tile_type)
        # Create a channel with the count value
        hand_channel = torch.full((1, height, width), count / 3, dtype=torch.float)
        hand_channel *= valid_positions_mask
        hand_channels.append(hand_channel)
    
    # Concatenate all channels
    channels_to_concat = [tensor, player_channel, phase_channel]
    if pile_channels:
        channels_to_concat.extend(pile_channels)
    if hand_channels:
        channels_to_concat.extend(hand_channels)
    
    full_tensor = torch.cat(channels_to_concat, dim=0)
    
    return full_tensor


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
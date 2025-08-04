# Game engine configuration
# Set GAME_TYPE to either "harmonies" or "connect4"
GAME_TYPE = "connect4"

# Import the appropriate game engine and state processor based on GAME_TYPE
if GAME_TYPE == "harmonies":
    from harmonies_engine import HarmoniesGameState as GameState
    from process_game_state import (
        create_state_tensors,
        create_board_tensor,
        create_global_features,
        get_action_index,
        get_action_from_index,
    )
    # Harmonies-specific constants
    ACTION_SIZE = 276  # This should match your Harmonies action space
    BOARD_TENSOR_SHAPE = (38, 5, 7)  # Channels, Height, Width for Harmonies
    GLOBAL_FEATURES_SIZE = 6  # Number of global features for Harmonies
    
    def get_action_size():
        return ACTION_SIZE
    
elif GAME_TYPE == "connect4":
    from connect4_engine import Connect4GameState as GameState
    from process_connect4_state import (
        create_state_tensors,
        create_board_tensor,
        create_global_features,
        get_action_index,
        get_action_from_index,
        get_action_size,
    )
    # Connect4-specific constants
    ACTION_SIZE = get_action_size()  # 7 for Connect4
    BOARD_TENSOR_SHAPE = (3, 6, 7)  # Channels, Height, Width for Connect4
    GLOBAL_FEATURES_SIZE = 2  # Number of global features for Connect4
    
else:
    raise ValueError(f"Unknown game type: {GAME_TYPE}")

# Export the GameState class with a consistent name
__all__ = ['GameState', 'create_state_tensors', 'get_action_index', 
           'ACTION_SIZE', 'BOARD_TENSOR_SHAPE', 'GLOBAL_FEATURES_SIZE']
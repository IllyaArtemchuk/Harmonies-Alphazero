import numpy as np
import torch
from typing import Tuple

def create_state_tensors(game_state) -> Tuple[torch.Tensor, torch.Tensor]:
    return (create_board_tensor(game_state), create_global_features(game_state))

def create_board_tensor(game_state) -> torch.Tensor:
    """
    Creates a spatial tensor representing Connect 4 board state.
    Output shape: (C, H, W) = (3, 6, 7)
    
    Channels:
    - 0: Current player's pieces (1 where current player has a piece)
    - 1: Opponent's pieces (1 where opponent has a piece)
    - 2: All ones (constant plane to help convolutions detect board edges)
    """
    tensor = torch.zeros(3, 6, 7, dtype=torch.float32)
    
    current_player = game_state.current_player
    opponent = 1 - current_player
    
    for row in range(6):
        for col in range(7):
            if game_state.board[row, col] == current_player + 1:
                tensor[0, row, col] = 1.0
            elif game_state.board[row, col] == opponent + 1:
                tensor[1, row, col] = 1.0
    
    tensor[2, :, :] = 1.0
    
    return tensor

def create_global_features(game_state) -> torch.Tensor:
    """
    Creates global features vector.
    Output shape: (2,)
    
    Features:
    - 0: Number of moves played / 42 (normalized)
    - 1: Current player (0 or 1)
    """
    features = torch.zeros(2, dtype=torch.float32)
    
    moves_played = len(game_state.move_history)
    features[0] = moves_played / 42.0
    features[1] = float(game_state.current_player)
    
    return features

def get_action_index(action: int) -> int:
    """
    Convert column index to action index for neural network.
    In Connect 4, actions are simply column indices (0-6).
    """
    return action

def get_action_from_index(index: int) -> int:
    """
    Convert action index from neural network to column index.
    """
    return index

def get_action_size() -> int:
    """
    Returns the size of the action space for Connect 4.
    """
    return 7
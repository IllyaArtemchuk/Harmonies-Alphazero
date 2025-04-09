from config import *
from harmonies_engine import HarmoniesGameState
from process_game_state import create_state_tensors
from MCTS import get_best_action_and_pi

class Trainer():
    def __init__(self):
        pass

def run_self_play_game(model_manager, mcts_config, self_play_config):
    """
    Plays one full game using AlphaZero MCTS and collects training examples.

    Args:
        model_manager: The ModelManager instance.
        mcts_config: Dictionary with MCTS hyperparameters.
        self_play_config: Dictionary with self-play hyperparameters.

    Returns:
        list: A list of training examples for this game, where each example is a tuple:
              (board_tensor, global_features_tensor, pi_target_tensor, outcome_perspective_tensor).
              Returns empty list if a significant error occurs during the game.
    """
    game = HarmoniesGameState() # Start new game
    # Store history entries as dictionaries for clarity
    game_history = [] # Stores {'board_rep': tensor, 'global_rep': tensor, 'player': int, 'pi': ndarray}

    while not game.is_game_over():
        current_player_idx = game.get_current_player() # Get the player index (0 or 1)

        # --- Generate NN Inputs (State Representation) ---
        # Calculate these BEFORE the MCTS search for the current state
        try:
            state_tensors = create_state_tensors(game)
            state_tensors = tuple(item.float() for item in state_tensors) # Ensure Float

        except Exception as e:
             print(f"ERROR: Failed to generate state representation for NN: {e}")
             print(f"State:\n{game}")
             return [] # Abort game if representation fails

        # --- Run MCTS Search ---
        try:
            # Pass the current game state (clone is good practice), model_manager, and configs
            best_action, pi_target = get_best_action_and_pi(
                game.clone(), # Pass a clone for safety during search
                model_manager, 
                mcts_config, 
                self_play_config
            ) 
        except Exception as e:
            print(f"ERROR: Exception during MCTS search (get_best_action_and_pi): {e}")
            print(f"State:\n{game}")
            return [] # Abort game

        # --- Handle MCTS Failure ---
        if best_action is None:
            print(f"WARNING: MCTS failed to return a valid action for player {current_player_idx}. Aborting game.")
            print(f"State:\n{game}")
            # Return empty list as this game's data might be unreliable
            return [] 
        
        # --- Store History (BEFORE applying move) ---
        # Store the NN inputs, the player, and the MCTS policy result
        game_history.append({
            'state_rep': state_tensors, # Store the tuple (board_tensor, global_features)
            'player': current_player_idx, 
            'pi': pi_target # pi_target should be a numpy array from MCTS
        })

        # --- Apply Move ---
        try:
            game = game.apply_move(best_action) 
        except Exception as e:
            print(f"ERROR: Exception during game.apply_move: {e}. Aborting game.")
            # Log state *before* the failed move
            # (Access state from the last 'state_representation' or re-generate if needed)
            print(f"Action attempted: {best_action}")
            return [] # Return empty list for this failed game

    # Game over
    final_outcome = game.get_game_outcome()

    if final_outcome is None: # Should not happen if game_is_over is true
         print("ERROR: Game ended but get_game_outcome returned None!")
         return []

    # --- Process Game History into Final Training Data ---
    training_data = []
    for history_entry in game_history:
        s_board, s_global = history_entry['state_rep'] # Unpack the stored state representation
        pi_target_np = history_entry['pi']
        player_turn = history_entry['player']

        # Determine outcome z from the perspective of player_turn
        if final_outcome == 0: # Draw
            outcome_perspective = 0.0
        elif player_turn == 0: # It was Player 0's turn
            outcome_perspective = float(final_outcome) # 1.0 if P0 won, -1.0 if P1 won
        else: # It was Player 1's turn
            outcome_perspective = -float(final_outcome) # -1.0 if P0 won, 1.0 if P1 won

        # Append final training tuple, converting numpy pi and scalar outcome to tensors
        training_data.append((
            s_board,                                     # Already a tensor
            s_global,                                    # Already a tensor
            torch.tensor(pi_target_np, dtype=torch.float), # Convert pi numpy array to tensor
            torch.tensor([outcome_perspective], dtype=torch.float) # Outcome as single-element tensor
        ))

    return training_data
import torch
import numpy as np
import random # For human player's random choices if needed

from harmonies_engine import HarmoniesGameState, VALID_HEXES, TILE_TYPES
from model import ModelManager
from MCTS import get_best_action_and_pi # Assuming this is your AI's move function
from config import model_config_default, training_config_default, mcts_config_eval, self_play_config_default
from process_game_state import get_action_index, create_state_tensors # For displaying policy
from constants import NUM_PILES, NUM_HEXES, coordinate_to_index_map # For policy interpretation

# --- Helper Functions ---
def print_board(board_dict, player_id, board_size_q_r=(-3, 3, -2, 2)):
    """Prints a representation of one player's board."""
    q_min, q_max, r_min, r_max = board_size_q_r
    grid = {} # Using a dict for sparse grid

    # Populate grid with actual tiles
    for (q, r), stack in board_dict.items():
        grid[(q, r)] = "/".join(s[:1].upper() for s in stack) # e.g., W/S/P

    print(f"\n--- Player {player_id}'s Board ---")
    # For a 5-4-5-4-5 structure (approximate console rendering)
    # This is a simplified rendering; true hex grid alignment is hard in console
    indent_map = {-2: 0, -1: 1, 0: 0, 1: 1, 2: 0} # Rough indent for visual
    for r in range(r_min, r_max + 1):
        row_str = " " * (indent_map.get(r,0) * 2)
        for q in range(q_min, q_max + 1):
            if (q, r) in VALID_HEXES:
                tile_str = grid.get((q, r), ".") # '.' for empty
                row_str += f"[{tile_str:<3}]" # Fixed width for alignment
            else:
                row_str += "     " # Empty space for non-valid hexes
        print(row_str)
    print("-" * 20)


def get_human_action(game_state: HarmoniesGameState, legal_moves):
    """Gets action from human player."""
    if not legal_moves:
        print("No legal moves available for human!")
        return None

    print("\nYour turn. Legal moves:")
    if game_state.turn_phase == "choose_pile":
        for i, move_idx in enumerate(legal_moves):
            # Display actual piles if possible
            pile_content = game_state.available_piles[move_idx]
            print(f"  {i+1}: Choose Pile {move_idx} ({'/'.join(pile_content)})")
        while True:
            try:
                choice = int(input(f"Enter choice (1-{len(legal_moves)}): ")) - 1
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice] # Return the pile index
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    elif game_state.turn_phase.startswith("place_tile"):
        print(f"Tiles in hand: {game_state.tiles_in_hand}")
        # legal_moves are (tile_type, coord)
        for i, (tile_type, coord) in enumerate(legal_moves):
            print(f"  {i+1}: Place {tile_type.upper()} at {coord}")
        while True:
            try:
                choice = int(input(f"Enter choice (1-{len(legal_moves)}): ")) - 1
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice] # Return (tile_type, coord)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return None # Should not reach here

def display_ai_stats(policy_probs, value_pred, game_state: HarmoniesGameState, action_size):
    """Displays AI's thoughts."""
    print(f"\n--- AI Stats ---")
    print(f"AI Value Prediction (for current player {game_state.current_player}): {value_pred:.3f}")

    print("Top 5 Policy Probabilities:")
    # Create a reverse map from index to action description for readability
    # This is a simplified reverse map; a full one is more complex
    action_descriptions = {}
    for i in range(NUM_PILES): # Pile actions
        action_descriptions[i] = f"Choose Pile {i}"

    idx_to_coord = {v: k for k, v in coordinate_to_index_map.items()}
    for tile_idx, tile_type in enumerate(TILE_TYPES):
        for coord_idx_flat in range(NUM_HEXES): # 0-22
            action_idx = NUM_PILES + (tile_idx * NUM_HEXES) + coord_idx_flat
            coord = idx_to_coord.get(coord_idx_flat, f"InvalidCoordIdx{coord_idx_flat}")
            action_descriptions[action_idx] = f"Place {tile_type.upper()} at {coord}"

    # Get indices of top N probabilities
    top_n = 5
    if policy_probs is not None and len(policy_probs) > 0:
        # Ensure policy_probs is a flat numpy array
        if not isinstance(policy_probs, np.ndarray):
            try:
                policy_probs = np.array(policy_probs, dtype=float) # Explicitly cast to float
            except Exception as e:
                print(f"  Error converting policy_probs to numpy array: {e}")
                policy_probs = np.array([]) # Default to empty array on error

        if policy_probs.ndim > 1: # If it's still not flat (e.g. array of lists)
            try:
                # Attempt to flatten robustly, e.g. if it became an object array
                if policy_probs.dtype == 'object':
                     # This case is tricky, might need specific handling if it occurs
                     print("  Warning: policy_probs is an object array, attempting to flatten.")
                     # A simple flatten might not work as expected if it's truly ragged.
                     # For now, we'll try a standard flatten.
                     policy_probs = np.concatenate(policy_probs).ravel() if policy_probs.size > 0 else np.array([])
                else:
                    policy_probs = policy_probs.flatten()
            except Exception as e:
                print(f"  Error flattening policy_probs array: {e}")
                policy_probs = np.array([])


        # Check if policy_probs is empty or not 1D after conversion
        if policy_probs.ndim != 1 or policy_probs.size == 0:
            print("  Error: Policy probabilities are not in expected format.")
            return

        # Only consider legal moves for "top policies" to be meaningful
        # This requires mapping AI's legal moves to their policy indices
        # For simplicity now, just show top raw policy values. A better way:
        # legal_ai_moves = game_state.get_legal_moves()
        # legal_policy_indices = {get_action_index(m): m for m in legal_ai_moves}
        # top_indices = sorted(legal_policy_indices.keys(), key=lambda i: policy_probs[i], reverse=True)[:top_n]
        
        # Simpler version: just top overall policy predictions
        top_indices = np.argsort(policy_probs)[-top_n:][::-1]


        for i, idx in enumerate(top_indices):
            if 0 <= idx < action_size:
                desc = action_descriptions.get(idx, f"Unknown Action Index {idx}")
                prob = policy_probs[idx]
                print(f"  {i+1}. {desc}: {prob:.4f}")
            else:
                print(f"  {i+1}. Invalid action index {idx} found in top policy.")
    else:
        print("  Policy probabilities not available or empty.")
    print("-" * 18)


# --- Main Game Loop ---
if __name__ == "__main__":
    print("--- Harmonies: Human vs AI ---")

    # 1. Load AI Model
    print("Loading AI model...")
    ai_model_manager = ModelManager(model_config_default, training_config_default)
    # Use the path from your training config
    checkpoint_folder = self_play_config_default["checkpoint_folder"]
    best_model_filename = self_play_config_default.get("best_model_filename", "best_model.pth.tar")
    loaded = ai_model_manager.load_checkpoint(folder=checkpoint_folder, filename=best_model_filename)
    if not loaded:
        print(f"ERROR: Could not load AI model from {checkpoint_folder}/{best_model_filename}. Exiting.")
        exit()
    ai_model_manager.model.eval() # Ensure model is in eval mode
    print("AI Model loaded.")

    # MCTS config for AI (deterministic evaluation)
    ai_mcts_config = mcts_config_eval.copy() # Use your eval config
    # Ensure action_size is correct
    ai_mcts_config["action_size"] = model_config_default["action_size"]


    # 2. Game Setup
    game = HarmoniesGameState()
    human_player_id = -1
    while human_player_id not in [0, 1]:
        try:
            human_player_id = int(input("Do you want to be Player 0 (starts) or Player 1? Enter 0 or 1: "))
        except ValueError:
            print("Invalid input.")
    ai_player_id = 1 - human_player_id
    print(f"You are Player {human_player_id}. AI is Player {ai_player_id}.")

    game_move_count = 0 # Initialize game move counter
    # 3. Main Game Loop
    while not game.is_game_over():
        current_player = game.get_current_player()
        print(f"\n===== Turn: {game.turn_phase}, Current Player: {current_player} =====")

        # Display boards
        print_board(game.player_boards[0], 0)
        print_board(game.player_boards[1], 1)

        # Display game info
        print(f"Available Piles: {game.available_piles}")
        if game.turn_phase.startswith("place_tile"):
             print(f"Player {current_player}'s Hand: {game.tiles_in_hand}")
        print(f"Bag counts: {dict(sorted(game.tile_bag.items()))}")


        legal_moves = game.get_legal_moves()
        if not legal_moves:
            print(f"Player {current_player} has no legal moves! This might be an error or end of phase.")
            # This part needs careful handling based on game rules if it's a valid state
            # For now, assume it might be a bug or an unhandled game state.
            if game.turn_phase.startswith("place_tile") and not game.tiles_in_hand:
                print("  Hand is empty, trying to advance turn logic (simulating _end_turn_actions effect)")
                # This is a hacky way to try and recover if a player is stuck with no tiles
                # and the phase expects placement.
                if game.turn_phase == "place_tile_1": game.turn_phase = "place_tile_2"
                elif game.turn_phase == "place_tile_2": game.turn_phase = "place_tile_3"
                elif game.turn_phase == "place_tile_3": game._end_turn_actions() # Call private for simplicity here
                continue # Restart loop for the new state/player
            else:
                break # End game if truly stuck

        chosen_action = None
        raw_policy_probs = None # For AI
        value_prediction = None # For AI

        if current_player == human_player_id:
            chosen_action = get_human_action(game, legal_moves)
            if chosen_action is None: # Human failed to make a move
                print("Error with human move. Game ending.")
                break
        else: # AI's turn
            print(f"AI (Player {ai_player_id}) is thinking...")
            # Get AI's action and stats
            # The get_best_action_and_pi function also returns pi_target (policy)
            # We also need the direct value prediction from the network for the root state
            
            # Get raw policy and value for current state *before* MCTS search for display
            # This shows the network's "prior" belief
            current_board_tensor, current_global_tensor = create_state_tensors(game)
            raw_policy_logits, value_prediction = ai_model_manager.model(
                current_board_tensor.unsqueeze(0).to(ai_model_manager.device),
                current_global_tensor.unsqueeze(0).to(ai_model_manager.device)
            )
            raw_policy_probs = torch.softmax(raw_policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
            value_prediction = value_prediction.item() # Get scalar

            # Now get the MCTS refined action
            chosen_action, mcts_pi_target = get_best_action_and_pi(
                game.clone(), # Pass a clone for MCTS
                ai_model_manager,
                ai_mcts_config,
                game_move_count # Pass the current game move count
            )
            if chosen_action is None:
                print("AI failed to choose an action! Game ending.")
                break
            
            print(f"AI chose action: {chosen_action}")
            # Display AI stats using the raw policy from the network's direct prediction
            # MCTS pi_target is for training, raw_policy_probs is the network's "thought"
            display_ai_stats(raw_policy_probs, value_prediction, game, ai_mcts_config["action_size"])


        # Apply the chosen move
        try:
            game = game.apply_move(chosen_action)
        except Exception as e:
            print(f"ERROR applying move {chosen_action}: {e}")
            import traceback
            traceback.print_exc()
            break # End game on error
        
        game_move_count +=1 # Increment after a successful move by either player

    # 4. Game Over
    print("\n========== GAME OVER ==========")
    final_outcome = game.get_game_outcome() # Should be 1 for P0 win, -1 for P1 win, 0 for draw
    scores = game.final_scores

    print_board(game.player_boards[0], 0)
    print_board(game.player_boards[1], 1)
    print(f"Final Scores: Player 0: {scores[0]}, Player 1: {scores[1]}")

    if final_outcome == 1: # P0 won
        winner_is_human = (0 == human_player_id)
        print(f"Player 0 wins!")
    elif final_outcome == -1: # P1 won
        winner_is_human = (1 == human_player_id)
        print(f"Player 1 wins!")
    else: # Draw
        winner_is_human = False # Or handle differently
        print("It's a draw!")

    if current_player == human_player_id and final_outcome is not None : # Check if outcome is valid
        if (final_outcome == 1 and human_player_id == 0) or \
           (final_outcome == -1 and human_player_id == 1):
            print("Congratulations, you won!")
        elif final_outcome != 0:
             print("The AI won. Better luck next time!")
    elif final_outcome is None:
        print("Game ended inconclusively or due to an error.")
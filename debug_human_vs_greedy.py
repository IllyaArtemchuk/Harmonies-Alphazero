import torch
import numpy as np
import random

# Add project root to sys.path if this file is in a subfolder like GUI/
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_types import MCTSConfigType
from harmonies_engine import HarmoniesGameState, VALID_HEXES, TILE_TYPES
from model import ModelManager # Your AlphaZero Model
from evaluation import choose_move_greedy # Your Greedy Agent
from config import model_config_default, training_config_default, self_play_config_default, mcts_config_eval
# Ensure process_game_state provides the CORRECT get_action_index
from MCTS import get_best_action_and_pi # The core MCTS function
from config import model_config_default, training_config_default, self_play_config_default, mcts_config_eval # Need eval config
from process_game_state import get_action_index, create_state_tensors
from constants import NUM_PILES, NUM_HEXES, TILE_TYPES, coordinate_to_index_map, INITIAL_BAG, PILE_SIZE # Added constants


# --- Helper Functions ---
def print_board(board_dict, player_id, board_size_q_r=(-3, 3, -2, 2)):
    """Prints a representation of one player's board."""
    q_min, q_max, r_min, r_max = board_size_q_r
    grid = {}
    for (q, r), stack in board_dict.items():
        # Limit displayed stack string length for neatness
        stack_str = "/".join(s[:1].upper() for s in stack)
        if len(stack_str) > 5: # e.g., > 3 chars + 2 slashes
            stack_str = stack_str[:4] + "*"
        grid[(q, r)] = stack_str

    print(f"\n--- Player {player_id}'s Board ---")
    # Simple rectangular print (doesn't perfectly show hex adjacency)
    header = "q=  " + " ".join([f"{q:3d}" for q in range(q_min, q_max + 1)])
    print(header)
    print("r:")
    for r in range(r_min, r_max + 1):
        row_str = f"{r:2d}  "
        for q in range(q_min, q_max + 1):
            if (q, r) in VALID_HEXES:
                tile_str = grid.get((q, r), ".")
                row_str += f"[{tile_str:<3}]" # Pad to 3 chars inside brackets
            else:
                row_str += "     " # Empty space for non-valid hexes
        print(row_str)
    print("-" * len(header))


def get_human_action(game_state: HarmoniesGameState, legal_moves):
    """Gets action from human player."""
    if not legal_moves:
        print("No legal moves available for human!")
        return None

    print("\nYour turn. Legal moves:")
    if game_state.turn_phase == "choose_pile":
        for i, move_idx in enumerate(legal_moves):
            if 0 <= move_idx < len(game_state.available_piles):
                 pile_content = game_state.available_piles[move_idx]
                 print(f"  {i+1}: Choose Pile {move_idx} ({'/'.join(t[:3] for t in pile_content)})") # Show short tile names
            else:
                 # This case should be prevented by game engine returning valid indices
                 print(f"  {i+1}: Error - Invalid pile index {move_idx} provided as legal.")
                 continue # Skip displaying invalid option
        while True:
            try:
                choice_str = input(f"Enter choice number (1-{len(legal_moves)}) or 'q' to quit: ")
                if choice_str.lower() == 'q': return None # Allow quitting
                choice = int(choice_str) - 1
                if 0 <= choice < len(legal_moves):
                    # Ensure the chosen index corresponds to a valid move in the list
                    # (This is needed if the legal_moves list itself might be sparse or weird)
                    # In this case, legal_moves directly contains the valid pile indices
                    return legal_moves[choice]
                else:
                    print("Invalid choice number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    elif game_state.turn_phase.startswith("place_tile"):
        print(f"Tiles in hand: {', '.join(t.upper() for t in game_state.tiles_in_hand)}")
        # Ensure legal_moves are (tile_type, coord) tuples
        valid_display_moves = []
        for i, move in enumerate(legal_moves):
             if isinstance(move, tuple) and len(move) == 2 and isinstance(move[0], str) and isinstance(move[1], tuple):
                 tile_type, coord = move
                 print(f"  {i+1}: Place {tile_type.upper()} at {coord}")
                 valid_display_moves.append(move)
             else:
                 print(f"  {i+1}: Error - Invalid move format in legal_moves: {move}") # Should not happen

        if not valid_display_moves:
             print("Error: No valid displayable moves found, though legal_moves list was not empty.")
             return None # Indicate an issue

        while True:
            try:
                choice_str = input(f"Enter choice number (1-{len(valid_display_moves)}) or 'q' to quit: ")
                if choice_str.lower() == 'q': return None # Allow quitting
                choice = int(choice_str) - 1
                if 0 <= choice < len(valid_display_moves):
                    return valid_display_moves[choice] # Return the chosen (tile_type, coord) tuple
                else:
                    print("Invalid choice number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else: # Should not happen in normal play
        print(f"Warning: get_human_action called during unexpected phase: {game_state.turn_phase}")
        return None


def display_az_model_predictions(az_model_manager: ModelManager, game_state_to_analyze: HarmoniesGameState):
    """
    Gets predictions from the AlphaZero model for the given state and displays them.
    Assumes the state provided is the one the *next* player will face.
    """
    current_player_for_pred = game_state_to_analyze.current_player
    print(f"\n--- AlphaZero Model's Analysis (for Player {current_player_for_pred}'s upcoming turn) ---")

    # Check if model loaded
    if az_model_manager is None:
        print("  Model not loaded. Cannot provide predictions.")
        return

    try:
        board_tensor, global_tensor = create_state_tensors(game_state_to_analyze)

        # Ensure tensors are on the correct device and have batch dimension
        device = az_model_manager.device
        board_tensor = board_tensor.unsqueeze(0).to(device)
        global_tensor = global_tensor.unsqueeze(0).to(device)

        az_model_manager.model.eval() # Ensure eval mode
        with torch.no_grad():
            policy_logits, value_pred_tensor = az_model_manager.model(board_tensor, global_tensor)
            policy_probs_np = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value_pred = value_pred_tensor.item()
    except Exception as e:
        print(f"  Error during model prediction: {e}")
        return

    print(f"Model's Value Prediction: {value_pred:.4f} (Positive means good for Player {current_player_for_pred})")
    print(f"Model's Policy Predictions (Top 10) for Player {current_player_for_pred}:")

    action_descriptions = {}
    # Pile actions
    for i in range(NUM_PILES):
        action_descriptions[i] = f"Choose Pile {i}"
    # Placement actions
    idx_to_coord = {v: k for k, v in coordinate_to_index_map.items()} # Reverse map
    for tile_type_idx, tile_type_str in enumerate(TILE_TYPES):
        for coord_flat_idx in range(NUM_HEXES):
            coord_tuple = idx_to_coord.get(coord_flat_idx, f"?{coord_flat_idx}?")
            action_idx_model = NUM_PILES + (tile_type_idx * NUM_HEXES) + coord_flat_idx
            if 0 <= action_idx_model < len(policy_probs_np): # Check bounds
                 action_descriptions[action_idx_model] = f"Place {tile_type_str[:3].upper()} at {coord_tuple}"

    top_n = 10
    if len(policy_probs_np) < top_n: # Handle cases where policy vector might be smaller than expected
        top_n = len(policy_probs_np)

    top_indices = np.argsort(policy_probs_np)[-top_n:][::-1]

    # Get actual legal moves for the state being analyzed to compare
    actual_legal_moves = game_state_to_analyze.get_legal_moves()
    legal_move_indices = set()
    if actual_legal_moves:
        for move in actual_legal_moves:
            try:
                # Use the globally available get_action_index
                legal_move_indices.add(get_action_index(move))
            except ValueError:
                pass # Ignore if a legal move can't be indexed (indicates mismatch)


    for i, model_action_idx in enumerate(top_indices):
        # Check if index is valid for the policy array length
        if 0 <= model_action_idx < len(policy_probs_np):
            desc = action_descriptions.get(model_action_idx, f"Unknown Action Index {model_action_idx}")
            prob = policy_probs_np[model_action_idx]
            is_legal_marker = " (*)" if model_action_idx in legal_move_indices else ""
            print(f"  {i+1}. {desc:<35}: {prob:.4f}{is_legal_marker}") # Pad description
        else:
            print(f"  {i+1}. Error - Invalid action index {model_action_idx} in top policy list.")

    if len(top_indices) > 0 : print("  (*) indicates the move is currently legal.")
    else: print("  No policy predictions found.")

    print("\n--- Full Policy Vector ---")
    # Print non-zero probabilities or format nicely
    formatted_probs = [f"{p:.2e}" if p > 1e-9 else "0.00e+00" for p in policy_probs_np]
    # Print in groups for readability
    GROUP_SIZE = 10
    for i in range(0, len(formatted_probs), GROUP_SIZE):
         print(" ".join(formatted_probs[i:i+GROUP_SIZE]))

    print("-" * 30)
    
def display_mcts_analysis(az_model_manager: ModelManager, mcts_config: MCTSConfigType, game_state_to_analyze: HarmoniesGameState):
    """
    Runs MCTS search for the given state using the AZ model and displays the results.
    Assumes the state provided is the one the *next* player will face.
    """
    current_player_for_pred = game_state_to_analyze.current_player
    print(f"\n--- MCTS Analysis (for Player {current_player_for_pred}'s upcoming turn) ---")

    if az_model_manager is None:
        print("  Model not loaded. Cannot perform MCTS analysis.")
        return

    # --- Run MCTS Search ---
    print(f"Running MCTS ({mcts_config.get('num_simulations', '?')} simulations)...")
    start_time = time.time()
    game_move_number_for_analysis = 0 # Or derive from game_state if possible, for now 0
    try:
        # Make sure MCTS config has correct action size
        mcts_config['action_size'] = az_model_manager.model_config['action_size']

        # Run the full MCTS process
        mcts_chosen_action, mcts_pi_target = get_best_action_and_pi(
            game_state_to_analyze.clone(), # IMPORTANT: Pass a clone!
            az_model_manager,
            mcts_config, # Pass the specific config (e.g., mcts_config_eval)
            game_move_number_for_analysis # Pass a move number
        )
    except Exception as e:
        print(f"  Error during MCTS execution: {e}")
        import traceback
        traceback.print_exc()
        return
    end_time = time.time()
    print(f"MCTS search took {end_time - start_time:.2f} seconds.")
    # --- End MCTS Search ---

    if mcts_chosen_action is None:
        print("MCTS result: No action chosen (possibly no legal moves or error).")
        return

    print(f"MCTS Chosen Action: {mcts_chosen_action}")
    print(f"MCTS Policy (Top 10 based on visit counts) for Player {current_player_for_pred}:")

    action_descriptions = {}
    for i in range(NUM_PILES):
        action_descriptions[i] = f"Choose Pile {i}"
    idx_to_coord = {v: k for k, v in coordinate_to_index_map.items()}
    for tile_type_idx, tile_type_str in enumerate(TILE_TYPES):
        for coord_flat_idx in range(NUM_HEXES):
            coord_tuple = idx_to_coord.get(coord_flat_idx, f"?{coord_flat_idx}?")
            action_idx_model = NUM_PILES + (tile_type_idx * NUM_HEXES) + coord_flat_idx
            if 0 <= action_idx_model < mcts_config['action_size']: # Check bounds
                 action_descriptions[action_idx_model] = f"Place {tile_type_str[:3].upper()} at {coord_tuple}"

    top_n = 10
    if len(mcts_pi_target) < top_n:
        top_n = len(mcts_pi_target)

    top_indices = np.argsort(mcts_pi_target)[-top_n:][::-1]

    actual_legal_moves = game_state_to_analyze.get_legal_moves()
    legal_move_indices = set()
    if actual_legal_moves:
        for move in actual_legal_moves:
            try:
                legal_move_indices.add(get_action_index(move))
            except ValueError:
                pass

    for i, mcts_action_idx in enumerate(top_indices):
        # Check index validity
        if 0 <= mcts_action_idx < len(mcts_pi_target):
            desc = action_descriptions.get(mcts_action_idx, f"Unknown Action Index {mcts_action_idx}")
            prob = mcts_pi_target[mcts_action_idx] # This is normalized visit count
            # Only print if probability is non-negligible
            if prob > 1e-6:
                 is_legal_marker = " (*)" if mcts_action_idx in legal_move_indices else ""
                 print(f"  {i+1}. {desc:<35}: {prob:.4f}{is_legal_marker}")
        else:
             print(f"  {i+1}. Error - Invalid action index {mcts_action_idx} in top MCTS policy list.")

    if len(top_indices) > 0: print("  (*) indicates the move is currently legal.")
    else: print("  No MCTS policy results found.")

    print("\n--- Full MCTS Pi Target Vector (Visit Distribution) ---")
    formatted_probs = [f"{p:.2e}" if p > 1e-9 else "0.00e+00" for p in mcts_pi_target]
    GROUP_SIZE = 10
    for i in range(0, len(formatted_probs), GROUP_SIZE):
         print(" ".join(formatted_probs[i:i+GROUP_SIZE]))

    print("-" * 30)

    # --- Optional: Also show raw NN value for comparison ---
    print("--- Raw NN Value Prediction (for comparison) ---")
    try:
        board_tensor, global_tensor = create_state_tensors(game_state_to_analyze)
        device = az_model_manager.device
        board_tensor = board_tensor.unsqueeze(0).to(device)
        global_tensor = global_tensor.unsqueeze(0).to(device)
        az_model_manager.model.eval()
        with torch.no_grad():
            _, value_pred_tensor = az_model_manager.model(board_tensor, global_tensor)
            value_pred = value_pred_tensor.item()
        print(f"Raw NN Value: {value_pred:.4f} (For Player {current_player_for_pred})")
    except Exception as e:
        print(f"  Error getting raw NN value: {e}")
    print("-" * 30)

# --- Main Game Loop ---
if __name__ == "__main__":
    print("--- Harmonies: Human vs Greedy (with MCTS Analysis) ---")

    # 1. Load AlphaZero Model (Needed for MCTS)
    az_model_manager = None
    print("Loading AlphaZero model...")
    try:
        # Ensure training_config specifies the correct device
        az_model_manager = ModelManager(model_config_default, training_config_default)
        checkpoint_folder = self_play_config_default["checkpoint_folder"]
        best_model_filename = self_play_config_default.get("best_model_filename", "best_model.pth.tar")
        loaded = az_model_manager.load_checkpoint(folder=checkpoint_folder, filename=best_model_filename)
        if not loaded:
            print(f"WARNING: Could not load AZ model from {checkpoint_folder}/{best_model_filename}. MCTS analysis will use an uninitialized model.")
        else:
            print("AlphaZero Model loaded successfully.")
        az_model_manager.model.eval()
    except Exception as e:
        print(f"ERROR loading AlphaZero model: {e}. Cannot perform MCTS analysis.")
        az_model_manager = None # Ensure it's None


    # Use the evaluation MCTS config
    mcts_analysis_config = mcts_config_eval.copy()
    # Optional: Reduce simulations if analysis takes too long for interactive play
    # mcts_analysis_config['num_simulations'] = 50 # Example: lower sims for faster feedback


    # 2. Game Setup
    game = HarmoniesGameState()
    human_player_id = -1
    while human_player_id not in [0, 1]:
        try:
            choice_str = input("Play as Player 0 (starts) or Player 1? Enter 0 or 1: ")
            human_player_id = int(choice_str)
        except ValueError:
            print("Invalid input.")
    greedy_player_id = 1 - human_player_id
    print(f"You are Player {human_player_id}. Greedy AI is Player {greedy_player_id}.")

    # 3. Main Game Loop
    turn_counter = 0 # Simple turn counter
    while not game.is_game_over():
        current_player = game.get_current_player()
        turn_counter += 1
        print(f"\n===== Turn {turn_counter} | Phase: {game.turn_phase} | Player: {current_player} =====")

        # Display Boards and Game Info
        print_board(game.player_boards[0], 0)
        print_board(game.player_boards[1], 1)
        print(f"Available Piles: {game.available_piles}")
        if game.turn_phase.startswith("place_tile"):
             print(f"Player {current_player}'s Hand: {game.tiles_in_hand}")
        bag_total = sum(game.tile_bag.values())
        if bag_total > 0:
            print(f"Bag counts ({bag_total} total): {dict(sorted(game.tile_bag.items()))}")
        else:
            print("Bag is empty.")

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            print(f"Player {current_player} has NO LEGAL MOVES!")
            if not game.is_game_over():
                 print("Warning: No legal moves, but game state says not over. Check engine logic.")
            break

        chosen_action = None
        next_game_state = None

        if current_player == human_player_id:
            chosen_action = get_human_action(game, legal_moves)
            if chosen_action is None:
                print("Quitting game.")
                break

            try:
                next_game_state = game.apply_move(chosen_action)
            except Exception as e:
                print(f"\nERROR applying YOUR move {chosen_action}: {e}\n")
                import traceback
                traceback.print_exc()
                input("Press Enter to exit.")
                break

            # **** DISPLAY MCTS ANALYSIS FOR THE STATE THE OPPONENT WILL FACE ****
            if not next_game_state.is_game_over():
                if az_model_manager: # Check if model loaded
                    display_mcts_analysis(az_model_manager, mcts_analysis_config, next_game_state)
                else:
                    print("\n(Model not loaded, skipping MCTS analysis)")

        else: # Greedy AI's turn
            print(f"Greedy AI (Player {greedy_player_id}) is thinking...")
            greedy_action_tuple = choose_move_greedy(game.clone())

            if greedy_action_tuple and greedy_action_tuple[0] is not None:
                chosen_action = greedy_action_tuple[0]
                print(f"Greedy AI chose action: {chosen_action}")
                try:
                    next_game_state = game.apply_move(chosen_action)
                except Exception as e:
                    print(f"\nERROR applying GREEDY's move {chosen_action}: {e}\n")
                    import traceback
                    traceback.print_exc()
                    input("Press Enter to exit.")
                    break
            else:
                print("ERROR: Greedy AI returned None action, but legal moves existed?")
                break

        if next_game_state:
            game = next_game_state
        else:
            print("Error: Move was chosen but next game state wasn't generated.")
            break

        if not game.is_game_over():
             if current_player == human_player_id:
                  input("Press Enter to continue to Greedy AI's turn...")
             else:
                  print("\n----------------------------------------")
                  # No pause needed before human turn, loop will show board etc.

    # 4. Game Over (keep as before)
    print("\n================ GAME OVER ================")
    final_outcome = game.get_game_outcome()
    scores = game.final_scores

    print("--- Final Boards ---")
    print_board(game.player_boards[0], 0)
    print_board(game.player_boards[1], 1)
    print(f"\nFinal Scores: Player 0: {scores[0]}, Player 1: {scores[1]}")

    winner_msg = "Game ended."
    if final_outcome == 1:
        winner_msg = f"Player 0 wins!"
        if human_player_id == 0: winner_msg += " Congratulations, you won!"
        else: winner_msg += " The Greedy AI won."
    elif final_outcome == -1:
        winner_msg = f"Player 1 wins!"
        if human_player_id == 1: winner_msg += " Congratulations, you won!"
        else: winner_msg += " The Greedy AI won."
    elif final_outcome == 0 :
        winner_msg = "It's a draw!"
    else:
        winner_msg = "Game ended inconclusively or outcome was None."

    print(f"\n{winner_msg}\n")
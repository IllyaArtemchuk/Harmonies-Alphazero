import torch
import numpy as np
import random

# Add project root to sys.path if this file is in a subfolder like GUI/
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from harmonies_engine import HarmoniesGameState, VALID_HEXES, TILE_TYPES
from model import ModelManager # Your AlphaZero Model
from evaluation import choose_move_greedy # Your Greedy Agent
from config import model_config_default, training_config_default, self_play_config_default
from process_game_state import get_action_index, create_state_tensors
from constants import NUM_PILES, NUM_HEXES, TILE_TYPES, coordinate_to_index_map


# --- Helper Functions (can reuse from play_against_ai.py or redefine) ---
def print_board(board_dict, player_id, board_size_q_r=(-3, 3, -2, 2)):
    """Prints a representation of one player's board."""
    q_min, q_max, r_min, r_max = board_size_q_r
    grid = {}
    for (q, r), stack in board_dict.items():
        grid[(q, r)] = "/".join(s[:1].upper() for s in stack)

    print(f"\n--- Player {player_id}'s Board ---")
    indent_map = {-2: 0, -1: 1, 0: 0, 1: 1, 2: 0}
    for r in range(r_min, r_max + 1):
        row_str = " " * (indent_map.get(r,0) * 2)
        for q in range(q_min, q_max + 1):
            if (q, r) in VALID_HEXES:
                tile_str = grid.get((q, r), ".")
                row_str += f"[{tile_str:<3}]"
            else:
                row_str += "     "
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
            if 0 <= move_idx < len(game_state.available_piles):
                 pile_content = game_state.available_piles[move_idx]
                 print(f"  {i+1}: Choose Pile {move_idx} ({'/'.join(pile_content)})")
            else:
                 print(f"  {i+1}: Invalid pile index {move_idx} in legal_moves.") # Should not happen
        while True:
            try:
                choice = int(input(f"Enter choice (1-{len(legal_moves)}): ")) - 1
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    elif game_state.turn_phase.startswith("place_tile"):
        print(f"Tiles in hand: {game_state.tiles_in_hand}")
        for i, (tile_type, coord) in enumerate(legal_moves):
            print(f"  {i+1}: Place {tile_type.upper()} at {coord}")
        while True:
            try:
                choice = int(input(f"Enter choice (1-{len(legal_moves)}): ")) - 1
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return None

def display_az_model_predictions(az_model_manager: ModelManager, game_state_after_human_move: HarmoniesGameState):
    """
    Gets predictions from the AlphaZero model for the given state and displays them.
    The state is AFTER the human has moved, so it's now the opponent's (Greedy's) turn.
    The value prediction will be from Greedy's perspective.
    The policy will be for Greedy's possible next moves.
    """
    print(f"\n--- AlphaZero Model's Analysis (for Player {game_state_after_human_move.current_player}'s turn) ---")

    board_tensor, global_tensor = create_state_tensors(game_state_after_human_move)
    
    # Ensure tensors are on the correct device and have batch dimension
    device = az_model_manager.device
    board_tensor = board_tensor.unsqueeze(0).to(device)
    global_tensor = global_tensor.unsqueeze(0).to(device)

    az_model_manager.model.eval()
    with torch.no_grad():
        policy_logits, value_pred_tensor = az_model_manager.model(board_tensor, global_tensor)
        policy_probs_np = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value_pred = value_pred_tensor.item()

    print(f"Model's Value Prediction: {value_pred:.4f} (Positive means good for current player: {game_state_after_human_move.current_player})")

    print("Model's Top 5 Policy Predictions (for current player's next move):")
    
    action_descriptions = {}
    # Pile actions
    for i in range(NUM_PILES):
        action_descriptions[i] = f"Choose Pile {i}"
    # Placement actions
    idx_to_coord = {v: k for k, v in coordinate_to_index_map.items()} # Reverse map
    for tile_type_idx, tile_type_str in enumerate(TILE_TYPES):
        for coord_flat_idx in range(NUM_HEXES):
            coord_tuple = idx_to_coord.get(coord_flat_idx, f"InvalidCoord{coord_flat_idx}")
            action_idx_model = NUM_PILES + (tile_type_idx * NUM_HEXES) + coord_flat_idx
            action_descriptions[action_idx_model] = f"Place {tile_type_str.upper()} at {coord_tuple}"

    top_n = 10 # Show more for debugging
    top_indices = np.argsort(policy_probs_np)[-top_n:][::-1]

    # Get legal moves for the *current* state to highlight them
    # Note: these are legal moves for the *next* player (Greedy)
    actual_legal_moves = game_state_after_human_move.get_legal_moves()
    legal_move_indices = set()
    if actual_legal_moves: # Check if not empty
        for move in actual_legal_moves:
            try:
                legal_move_indices.add(get_action_index(move)) # Use your current get_action_index
            except ValueError: # If a legal move can't be indexed (should not happen)
                pass


    for i, model_action_idx in enumerate(top_indices):
        if 0 <= model_action_idx < len(policy_probs_np):
            desc = action_descriptions.get(model_action_idx, f"Unknown Action Index {model_action_idx}")
            prob = policy_probs_np[model_action_idx]
            is_legal_marker = "(*)" if model_action_idx in legal_move_indices else ""
            print(f"  {i+1}. {desc}: {prob:.4f} {is_legal_marker}")
        else:
            print(f"  {i+1}. Invalid action index {model_action_idx} from model policy.")
    print("(*) indicates the move is actually legal from this state for the current player.")
    print("-" * 30)
    
    print('---POLICY VECTOR---')
    print(policy_probs_np)


# --- Main Game Loop ---
if __name__ == "__main__":
    print("--- Harmonies: Human vs Greedy (with AZ Model Analysis) ---")

    # 1. Load AlphaZero Model (for analysis only)
    print("Loading AlphaZero model for analysis...")
    az_model_manager = ModelManager(model_config_default, training_config_default)
    checkpoint_folder = self_play_config_default["checkpoint_folder"]
    best_model_filename = self_play_config_default.get("best_model_filename", "best_model.pth.tar")
    loaded = az_model_manager.load_checkpoint(folder=checkpoint_folder, filename=best_model_filename)
    if not loaded:
        print(f"WARNING: Could not load AZ model from {checkpoint_folder}/{best_model_filename}. Analysis will be from an uninitialized model.")
    else:
        print("AlphaZero Model loaded for analysis.")
    az_model_manager.model.eval()

    # 2. Game Setup
    game = HarmoniesGameState()
    human_player_id = -1
    while human_player_id not in [0, 1]:
        try:
            human_player_id = int(input("Play as Player 0 (starts) or Player 1? Enter 0 or 1: "))
        except ValueError:
            print("Invalid input.")
    greedy_player_id = 1 - human_player_id
    print(f"You are Player {human_player_id}. Greedy AI is Player {greedy_player_id}.")

    # 3. Main Game Loop
    while not game.is_game_over():
        current_player = game.get_current_player()
        print(f"\n===== Turn: {game.turn_phase}, Current Player: {current_player} =====")

        print_board(game.player_boards[0], 0)
        print_board(game.player_boards[1], 1)
        print(f"Available Piles: {game.available_piles}")
        if game.turn_phase.startswith("place_tile"):
             print(f"Player {current_player}'s Hand: {game.tiles_in_hand}")
        print(f"Bag counts: {dict(sorted(game.tile_bag.items()))}")

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            print(f"Player {current_player} has no legal moves! This might be an error or end of phase.")
            # Attempt to advance phase if hand is empty during placement
            if game.turn_phase.startswith("place_tile") and not game.tiles_in_hand:
                print("  Hand is empty, trying to advance turn logic (simulating _end_turn_actions effect)")
                if game.turn_phase == "place_tile_1": game.turn_phase = "place_tile_2"
                elif game.turn_phase == "place_tile_2": game.turn_phase = "place_tile_3"
                elif game.turn_phase == "place_tile_3":
                    game._end_turn_actions() # Call private for simplicity here
                    if game.is_game_over(): break # Check if game ended after this
                continue
            else:
                break # End game if truly stuck

        chosen_action = None

        if current_player == human_player_id:
            chosen_action = get_human_action(game, legal_moves)
            if chosen_action is None:
                print("Error with human move. Game ending.")
                break
            
            # Apply human move
            try:
                game_after_human_move = game.apply_move(chosen_action)
            except Exception as e:
                print(f"ERROR applying YOUR move {chosen_action}: {e}")
                break
            
            # **** DISPLAY AZ MODEL PREDICTIONS FOR THE NEW STATE ****
            if not game_after_human_move.is_game_over(): # Only analyze if game isn't over
                display_az_model_predictions(az_model_manager, game_after_human_move)
            
            game = game_after_human_move # Commit the move

        else: # Greedy AI's turn
            print(f"Greedy AI (Player {greedy_player_id}) is thinking...")
            # Greedy agent's choose_move_greedy expects (game_state) and returns (action, ())
            # We need to pass a clone to greedy agent as it might simulate apply_move
            greedy_action_tuple = choose_move_greedy(game.clone())
            if greedy_action_tuple and greedy_action_tuple[0] is not None:
                chosen_action = greedy_action_tuple[0]
            else:
                print("Greedy AI failed to choose an action! (No legal moves or error). Game ending.")
                # If greedy has no moves, it might be a valid end-game state where current player has no options
                # or a bug in greedy/game state.
                if not game.get_legal_moves(): # If game state confirms no legal moves for greedy
                    print("Confirmed: Greedy has no legal moves. This may be part of game ending.")
                break # End game

            print(f"Greedy AI chose action: {chosen_action}")
            # Apply Greedy's move
            try:
                game = game.apply_move(chosen_action)
            except Exception as e:
                print(f"ERROR applying GREEDY's move {chosen_action}: {e}")
                break
        
        # Input to pause and review before AI's next move or game end
        if not game.is_game_over() and current_player == human_player_id : # After human turn and analysis
            input("Press Enter to continue to Greedy AI's turn...")


    # 4. Game Over
    print("\n========== GAME OVER ==========")
    final_outcome = game.get_game_outcome()
    scores = game.final_scores

    print_board(game.player_boards[0], 0)
    print_board(game.player_boards[1], 1)
    print(f"Final Scores: Player 0: {scores[0]}, Player 1: {scores[1]}")

    # Determine winner based on IDs
    if final_outcome == 1: # P0 won
        print(f"Player 0 wins!")
        if human_player_id == 0: print("Congratulations, you won!")
        else: print("The Greedy AI won.")
    elif final_outcome == -1: # P1 won
        print(f"Player 1 wins!")
        if human_player_id == 1: print("Congratulations, you won!")
        else: print("The Greedy AI won.")
    elif final_outcome == 0 : # Draw
        print("It's a draw!")
    else:
        print("Game ended inconclusively or due to an error.")
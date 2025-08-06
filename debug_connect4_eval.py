import torch
from model import ModelManager
from connect4_config import (
    connect4_model_config,
    connect4_training_config,
    connect4_mcts_config,
    connect4_self_play_config,
    connect4_mcts_eval_config,
)
from MCTS import get_best_action_and_pi
from connect4_evaluation import choose_move_greedy_connect4
from connect4_engine import Connect4GameState

def debug_play_connect4_game(player0_func, player1_func, args0=None, args1=None, show_moves=True):
    """
    Plays a single Connect 4 game with detailed debugging output.
    """
    game = Connect4GameState()
    players = {0: player0_func, 1: player1_func}
    player_args = {
        0: args0 if args0 is not None else (),
        1: args1 if args1 is not None else (),
    }
    
    player_names = {0: "Candidate Model", 1: "Best Model"}
    
    print("=== Starting Debug Game ===")
    game.render()
    
    move_count = 0
    while not game.is_game_over() and move_count < 42:
        current_player = game.get_current_player()
        move_function = players[current_player]
        current_args = player_args[current_player]
        
        print(f"\nMove {move_count + 1}: {player_names[current_player]} (Player {current_player}) to play")
        
        # Prepare arguments for the move function
        move_args = (game.clone(),) + current_args
        
        try:
            result = move_function(*move_args)
            if isinstance(result, tuple):  # AZ agent likely returned (action, pi)
                best_action = result[0]
                if len(result) > 1 and hasattr(result[1], '__len__'):
                    pi = result[1]
                    if len(pi) > 0:
                        print(f"  Policy values: {[f'{i}:{pi[i]:.3f}' for i in range(min(7, len(pi))) if pi[i] > 0.01]}")
            else:
                best_action = result
            
            if best_action is None:
                print(f"ERROR: {player_names[current_player]} returned None action.")
                return 0
            
            print(f"  {player_names[current_player]} chooses column: {best_action}")
            
        except Exception as e:
            print(f"ERROR: Exception during {player_names[current_player]}'s move: {e}")
            return 0
        
        # Apply the chosen move
        try:
            game = game.apply_move(best_action)
            move_count += 1
            
            if show_moves:
                game.render()
                
        except Exception as e:
            print(f"ERROR: Exception during apply_move: {e}")
            print(f"Action attempted: {best_action}")
            return 0
    
    # Game finished
    outcome = game.get_game_outcome()
    print(f"\n=== Game Finished ===")
    print(f"Final outcome: {outcome} (1=P0 wins, -1=P1 wins, 0=draw)")
    if outcome == 1:
        print(f"Winner: {player_names[0]} (Player 0)")
    elif outcome == -1:
        print(f"Winner: {player_names[1]} (Player 1)")
    else:
        print("Result: Draw or Error")
    
    return outcome

def debug_model_vs_model():
    """Debug candidate vs best model evaluation games."""
    print("=== Connect 4 Model vs Model Debug ===")
    
    # Load candidate model
    candidate_mgr = ModelManager(connect4_model_config, connect4_training_config)
    checkpoint_folder = connect4_self_play_config["checkpoint_folder"]
    
    # Try to load latest candidate
    candidate_loaded = candidate_mgr.load_checkpoint(
        folder=checkpoint_folder, filename="latest_candidate.pth.tar"
    )
    
    if not candidate_loaded:
        print("No candidate model found, using best model for both players")
        candidate_mgr.load_checkpoint(
            folder=checkpoint_folder, filename="best_model.pth.tar"
        )
    
    # Load best model
    best_mgr = ModelManager(connect4_model_config, connect4_training_config)
    best_loaded = best_mgr.load_checkpoint(
        folder=checkpoint_folder, filename="best_model.pth.tar"
    )
    
    if not best_loaded:
        print("ERROR: Could not load best model")
        return
    
    print(f"Loaded candidate: {candidate_loaded}, best: {best_loaded}")
    
    # Use proper evaluation config with more simulations
    eval_config = connect4_mcts_eval_config.copy()
    print(f"Using {eval_config['num_simulations']} MCTS simulations per move")
    
    # Arguments for both models
    candidate_args = (candidate_mgr, eval_config, 0)
    best_args = (best_mgr, eval_config, 0)
    
    # Play one debug game
    print("\n--- Game 1: Candidate (P0) vs Best (P1) ---")
    outcome1 = debug_play_connect4_game(
        get_best_action_and_pi, get_best_action_and_pi,
        candidate_args, best_args, show_moves=True
    )
    
    print("\n--- Game 2: Best (P0) vs Candidate (P1) ---")
    outcome2 = debug_play_connect4_game(
        get_best_action_and_pi, get_best_action_and_pi,
        best_args, candidate_args, show_moves=True
    )
    
    print(f"\n=== Summary ===")
    print(f"Game 1 (Candidate as P0): {outcome1}")
    print(f"Game 2 (Candidate as P1): {-outcome2}")  # Flip perspective
    
    candidate_wins = (1 if outcome1 == 1 else 0) + (1 if outcome2 == -1 else 0)
    best_wins = (1 if outcome1 == -1 else 0) + (1 if outcome2 == 1 else 0)
    draws = (1 if outcome1 == 0 else 0) + (1 if outcome2 == 0 else 0)
    
    print(f"Results: Candidate={candidate_wins}, Best={best_wins}, Draws={draws}")

def debug_model_vs_greedy():
    """Debug model vs greedy agent games."""
    print("=== Connect 4 Model vs Greedy Debug ===")
    
    # Load best model
    model_mgr = ModelManager(connect4_model_config, connect4_training_config)
    checkpoint_folder = connect4_self_play_config["checkpoint_folder"]
    
    loaded = model_mgr.load_checkpoint(
        folder=checkpoint_folder, filename="best_model.pth.tar"
    )
    
    if not loaded:
        print("ERROR: Could not load best model")
        return
    
    # Use evaluation config
    eval_config = connect4_mcts_eval_config.copy()
    print(f"Using {eval_config['num_simulations']} MCTS simulations per move")
    
    model_args = (model_mgr, eval_config, 0)
    
    print("\n--- Game 1: Model (P0) vs Greedy (P1) ---")
    outcome1 = debug_play_connect4_game(
        get_best_action_and_pi, choose_move_greedy_connect4,
        model_args, None, show_moves=True
    )
    
    print("\n--- Game 2: Greedy (P0) vs Model (P1) ---")
    outcome2 = debug_play_connect4_game(
        choose_move_greedy_connect4, get_best_action_and_pi,
        None, model_args, show_moves=True
    )
    
    print(f"\n=== Summary ===")
    print(f"Game 1 (Model as P0): {outcome1}")
    print(f"Game 2 (Model as P1): {-outcome2}")  # Flip perspective for model
    
    model_wins = (1 if outcome1 == 1 else 0) + (1 if outcome2 == -1 else 0)
    greedy_wins = (1 if outcome1 == -1 else 0) + (1 if outcome2 == 1 else 0)
    draws = (1 if outcome1 == 0 else 0) + (1 if outcome2 == 0 else 0)
    
    print(f"Results: Model={model_wins}, Greedy={greedy_wins}, Draws={draws}")

if __name__ == "__main__":
    print("Choose debug mode:")
    print("1. Model vs Model (training evaluation)")
    print("2. Model vs Greedy")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            debug_model_vs_model()
        elif choice == "2":
            debug_model_vs_greedy()
        else:
            print("Invalid choice, defaulting to Model vs Greedy")
            debug_model_vs_greedy()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        # Default to model vs greedy
        debug_model_vs_greedy() 
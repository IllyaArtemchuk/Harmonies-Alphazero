import torch
from model import ModelManager
from connect4_config import (
    connect4_model_config,
    connect4_training_config,
    connect4_mcts_config,
    connect4_self_play_config,
)
from MCTS import get_best_action_and_pi
from connect4_evaluation import choose_move_greedy_connect4, run_connect4_tournament

if __name__ == "__main__":

    NUM_EVAL_GAMES = 20  # How many games to play for evaluation

    print("--- Loading Best Connect 4 AlphaZero Model ---")
    # Use the Connect 4 specific configurations
    best_model_mgr = ModelManager(connect4_model_config, connect4_training_config)

    checkpoint_folder = connect4_self_play_config["checkpoint_folder"]
    best_model_filename = connect4_self_play_config.get(
        "best_model_filename", "best_model.pth.tar"
    )

    loaded = best_model_mgr.load_checkpoint(
        folder=checkpoint_folder, filename=best_model_filename
    )

    if not loaded:
        print(
            f"ERROR: Could not load best model from {checkpoint_folder}/{best_model_filename}"
        )
        print("Make sure you have trained a Connect 4 model first using main_connect4.py")
        exit()

    print(f"Successfully loaded model from {checkpoint_folder}/{best_model_filename}")

    # --- Prepare Arguments for Agents ---
    # Create a config for deterministic MCTS evaluation
    mcts_eval_config = connect4_mcts_config.copy()
    mcts_eval_config["dirichlet_epsilon"] = 0.0  # No noise for evaluation
    mcts_eval_config["num_simulations"] = min(400, connect4_mcts_config["num_simulations"])  # More simulations for stronger play

    # Arguments tuple for AlphaZero function
    az_arguments = (
        best_model_mgr,  # The loaded best model manager
        mcts_eval_config,  # Config for deterministic MCTS
        0  # Move count (starts at 0)
    )

    # Greedy agent doesn't need extra args beyond game_state
    greedy_arguments = None

    # --- Run the Tournament ---
    print(f"\nStarting Connect 4 evaluation with {NUM_EVAL_GAMES} games...")
    print(f"MCTS simulations per move: {mcts_eval_config['num_simulations']}")
    print(f"Using Connect 4 configurations from connect4_config.py")
    
    run_connect4_tournament(
        num_games=NUM_EVAL_GAMES,
        az_agent_func=get_best_action_and_pi,
        greedy_agent_func=choose_move_greedy_connect4,
        az_args=az_arguments,
        greedy_args=greedy_arguments,
    )

    print("\nConnect 4 evaluation against Greedy Agent complete.")
    print("\nNOTE: If you're still seeing unusual results, check:")
    print("1. That your model was actually trained on Connect 4 (using main_connect4.py)")
    print("2. That the checkpoint folder contains a Connect 4 model")
    print("3. That the game_config.py has GAME_TYPE = 'connect4'") 
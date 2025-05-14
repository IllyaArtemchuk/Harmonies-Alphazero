import torch  # For device check maybe
from model import ModelManager
from config import (
    model_config_default,
    training_config_default,
    mcts_config_default,
    self_play_config_default,
)
from MCTS import get_best_action_and_pi
from evaluation import choose_move_greedy, run_tournament  # Import the greedy function

if __name__ == "__main__":

    NUM_EVAL_GAMES = 20  # How many games to play for evaluation

    print("--- Loading Best AlphaZero Model ---")
    # Use the configs the BEST model was trained with (or compatible ones)
    # It's safer to load these from the checkpoint if you saved them!
    best_model_mgr = ModelManager(model_config_default, training_config_default)

    checkpoint_folder = self_play_config_default["checkpoint_folder"]
    best_model_filename = self_play_config_default.get(
        "best_model_filename", "best_model.pth.tar"
    )

    loaded = best_model_mgr.load_checkpoint(
        folder=checkpoint_folder, filename=best_model_filename
    )

    if not loaded:
        print(
            f"ERROR: Could not load best model from {checkpoint_folder}/{best_model_filename}"
        )
        exit()

    # --- Prepare Arguments for Agents ---
    # Create a config for deterministic MCTS evaluation
    mcts_eval_config = mcts_config_default.copy()
    mcts_eval_config["dirichlet_epsilon"] = 0.0  # No noise

    # Arguments tuple for AlphaZero function
    az_arguments = (
        best_model_mgr,  # The loaded best model manager
        mcts_eval_config,  # Config for deterministic MCTS
        0
    )

    # Greedy agent doesn't need extra args beyond game_state
    greedy_arguments = None

    # --- Run the Tournament ---
    run_tournament(
        num_games=NUM_EVAL_GAMES,
        az_agent_func=get_best_action_and_pi,
        greedy_agent_func=choose_move_greedy,
        az_args=az_arguments,
        greedy_args=greedy_arguments,
    )

    print("\nEvaluation against Greedy Agent complete.")

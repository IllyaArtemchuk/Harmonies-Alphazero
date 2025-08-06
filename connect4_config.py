"""
Configuration overrides for Connect 4 training.
These values are optimized for the simpler Connect 4 game.
"""

from config import (
    model_config_default,
    training_config_default,
    mcts_config_default,
    self_play_config_default
)

# Create Connect 4 specific configurations by copying and modifying defaults
connect4_model_config = model_config_default.copy()
connect4_model_config.update({
    "cnn_filters": 64,  # Reduced from 128 - Connect 4 is simpler
    "num_res_blocks": 4,  # Reduced from 8 - Connect 4 needs less depth
    "value_head_hidden_dim": 128,  # Reduced from 256
})

connect4_training_config = training_config_default.copy()
connect4_training_config.update({
    "learning_rate": 0.001,  # Good starting point
    "batch_size": 32,  # Smaller batch size for faster updates
})

connect4_mcts_config = mcts_config_default.copy()
connect4_mcts_config.update({
    "num_simulations": 200,  # Reduced from default - Connect 4 has smaller game tree
    "cpuct": 1.0,  # May need tuning
    "testing": False,  # IMPORTANT: Use False for proper evaluation with adequate simulations
})

# Create Connect 4 specific evaluation config with more simulations for stronger play
connect4_mcts_eval_config = {
    "num_simulations": 400,  # More simulations for stronger evaluation games
    "cpuct": 1.5,  # Slightly more exploration for evaluation
    "dirichlet_alpha": 0.1,
    "dirichlet_epsilon": 0.0,  # No noise for deterministic evaluation
    "fpu_value": 0.25,
    "turns_until_tau0": 0,  # Greedy move selection from start
    "action_size": connect4_model_config["action_size"],
    "testing": False,  # Use proper evaluation, not minimal testing config
}

connect4_self_play_config = self_play_config_default.copy()
connect4_self_play_config.update({
    "num_games_per_iter": 100,  # More games since they're faster
    "checkpoint_folder": "./checkpoints_connect4/",
    "replay_buffer_folder": "./buffer_connect4/",
    "eval_episodes": 20,  # Reasonable number for Connect 4
    "eval_win_rate_threshold": 0.55,  # Slightly higher threshold for Connect 4
})
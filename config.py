import torch
from constants import (
    ACTION_SIZE,
    BOARD_SIZE,
    INPUT_CHANNELS,
    GLOBAL_FEATURE_SIZE,
    coordinate_to_index_map,
    NUM_HEXES,
)
from config_types import (
    TrainingConfigType,
    ModelConfigType,
    MCTSConfigType,
    SelfPlayConfigType,
)


model_config_default: ModelConfigType = {
    # Parameters defining the NN architecture passed to AlphaZeroModel.__init__
    "input_channels": INPUT_CHANNELS,  # Channels from create_board_tensor (e.g., 38)
    "cnn_filters": 75,  # Filters in conv/residual blocks (CNN_FILTERS)
    "board_size": BOARD_SIZE,  # Tuple (H, W) of the spatial tensor (e.g., (5, 6))
    "action_size": ACTION_SIZE,  # Total size of the policy output vector (e.g., 143)
    "global_feature_size": GLOBAL_FEATURE_SIZE,  # Size of the global feature vector (e.g., 42)
    "value_head_hidden_dim": 256,  # Size of the hidden layer in the value head's MLP
    "num_res_blocks": 6,  # Number of residual blocks in the CNN body
    "policy_head_conv_filters": 2,  # Filters in the policy head's initial 1x1 conv
    "value_head_conv_filters": 1,  # Filters in the value head's initial 1x1 conv
}

training_config_default: TrainingConfigType = {
    # Parameters controlling the training optimization process passed to ModelManager.__init__
    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "optimizer_type": "Adam",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0001,  # L2 regularization strength
    "value_loss_weight": 0.5,
    "policy_loss_weight": 1.0,
    "batch_size": 64,
    # Note: EPOCHS = 1 from original seems to map to NUM_EPOCHS_PER_ITER in self_play_config
}

mcts_config_default: MCTSConfigType = {
    "num_simulations": 800,  # MCTS simulations per move
    "cpuct": 1,  # Exploration constant for PUCT
    # --- Parameters for Dirichlet noise added to root priors during self-play ---
    "dirichlet_alpha": 0.4,
    "dirichlet_epsilon": 0.2,
    "fpu_value": 0.25,
    # --- Temperature parameter for move selection ---
    "turns_until_tau0": 10,  # Turn after which move selection becomes deterministic
    # Before this turn, visits^(1/tau) is used, tau=1 usually.
    "action_size": model_config_default["action_size"],
    "testing": False,
}

mcts_config_eval: MCTSConfigType = {
    "num_simulations": 800,  # MCTS simulations per move
    "cpuct": 1,  # Exploration constant for PUCT
    # --- Parameters for Dirichlet noise added to root priors during self-play ---
    "dirichlet_alpha": 0.1,
    "dirichlet_epsilon": 0,
    "fpu_value": 0.25,
    # --- Temperature parameter for move selection ---
    "turns_until_tau0": 10,  # Turn after which move selection becomes deterministic
    # Before this turn, visits^(1/tau) is used, tau=1 usually.
    "action_size": model_config_default["action_size"],
    "testing": False,
}

self_play_config_default: SelfPlayConfigType = {
    "num_iterations": 100,  # Total number of self-play -> train iterations
    "num_games_per_iter": 25,  # Number of games generated per iteration
    "epochs_per_iter": 20,  # Number of training epochs over the buffer per iteration
    "num_parallel_games": 3,  # Number of games that will run in parallel
    "worker_device": "mps",  # Device used for the self play phase by the workers
    "replay_buffer_size": 50000,  # Max number of (s, pi, z) examples stored
    "checkpoint_folder": "./harmonies_az_run/",  # Folder to save model checkpoints
    "replay_buffer_folder": "./RUN_BUFFER/",
    "replay_buffer_filename": "replay_buffer.pkl",
    "best_model_filename": "best_model.pth.tar",
    # --- Evaluation Settings (run periodically, e.g., after N iterations) ---
    "eval_episodes": 20,  # Number of games to play between current and best model
    "eval_win_rate_threshold": 0.55,  # Win rate needed for new model to become the 'best'
    "eval_frequency": 10,  # How often evaluation is done (every N interations)
    # --- Info needed by helper functions ---
    "action_size": model_config_default["action_size"],
    "num_hexes": NUM_HEXES,
    "coordinate_to_index_map": coordinate_to_index_map,
}


### TESTING CONFIGS
test_model_config: ModelConfigType = {
    "input_channels": INPUT_CHANNELS,
    "cnn_filters": 32,  # Smaller filter size for faster NN pass (optional)
    "board_size": BOARD_SIZE,
    "action_size": ACTION_SIZE,
    "global_feature_size": GLOBAL_FEATURE_SIZE,
    "value_head_hidden_dim": 64,  # Smaller hidden dim for faster NN (optional)
    "num_res_blocks": 1,  # <<< Minimum residual blocks for speed
    "policy_head_conv_filters": 2,
    "value_head_conv_filters": 1,
}

# --- Training Config (Minimal training) ---
test_training_config: TrainingConfigType = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optimizer_type": "Adam",
    "learning_rate": 0.001,  # LR doesn't hugely impact test speed, keep standard
    "weight_decay": 0.0,  # Disable regularization for speed/simplicity in test
    "value_loss_weight": 1.0,  # Keep standard weights
    "policy_loss_weight": 1.0,
    "batch_size": 4,  # <<< VERY SMALL batch size
    "momentum": 0.9,
}

# --- MCTS Config (Minimal search) ---
test_mcts_config: MCTSConfigType = {
    "num_simulations": 4,  # <<< ABSOLUTE MINIMUM simulations
    "cpuct": 1.0,  # Keep standard exploration factor
    "dirichlet_alpha": 0.3,  # Noise params don't affect speed much
    "dirichlet_epsilon": 0.0,  # <<< DISABLE root noise for simplicity in test run
    "fpu_value": 0.25,
    "turns_until_tau0": 0,  # <<< Makes move selection greedy immediately (tau=0)
    # Add eval_mode flag if get_best_action_and_pi supports it
    "action_size": ACTION_SIZE,
    "testing": True,
}

test_mcts_config_eval: MCTSConfigType = {
    "num_simulations": 4,  # <<< ABSOLUTE MINIMUM simulations
    "cpuct": 1.0,  # Keep standard exploration factor
    "dirichlet_alpha": 0.1,
    "dirichlet_epsilon": 0.0,
    "fpu_value": 0.25,
    "turns_until_tau0": 0,  # <<< Makes move selection greedy immediately (tau=0)
    # Add eval_mode flag if get_best_action_and_pi supports it
    "action_size": ACTION_SIZE,
    "testing": True,
}

# --- Self-Play Config (Minimal execution) ---
test_self_play_config: SelfPlayConfigType = {
    "num_iterations": 1,  # <<< ONLY ONE iteration
    "num_games_per_iter": 2,  # <<< VERY FEW games
    "epochs_per_iter": 1,  # <<< Minimum training epochs
    "replay_buffer_size": 100,  # <<< Small buffer, just needs > batch_size*games
    "checkpoint_folder": "./TEST_RUN_CHECKPOINTS/",  # <<< Use a SEPARATE folder!
    "replay_buffer_folder": "./TEST_RUN_BUFFER/",  # <<< Use a SEPARATE folder!
    "replay_buffer_filename": "test_replay_buffer.pkl",
    # Evaluation - Disable by setting frequency > num_iterations
    "eval_frequency": 2,  # <<< Ensures evaluation doesn't run in 1 iteration
    "eval_episodes": 4,  # Lowered for faster potential eval testing later
    "eval_win_rate_threshold": 0.55,
    "best_model_filename": "test_best_model.pth.tar",  # Separate best model file
    # Parallelization - Use fewer workers for quick test
    "num_parallel_games": 1,  # <<< Low number, adjust based on your cores (e.g., max(1, cpu_count() // 2))
    "worker_device": "cpu",
    # --- Info needed by helper functions ---
    "action_size": test_model_config["action_size"],  # Reference from model config
    "num_hexes": NUM_HEXES,  # Make sure this matches constants.py
    "coordinate_to_index_map": coordinate_to_index_map,  # Make sure this matches constants.py
}

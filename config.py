import torch
from constants import (
    ACTION_SIZE,
    BOARD_SIZE,
    INPUT_CHANNELS,
    GLOBAL_FEATURE_SIZE,
    coordinate_to_index_map,
    NUM_HEXES,
)

model_config = {
    # Parameters defining the NN architecture passed to AlphaZeroModel.__init__
    "input_channels": INPUT_CHANNELS,  # Channels from create_board_tensor (e.g., 38)
    "cnn_filters": 75,  # Filters in conv/residual blocks (CNN_FILTERS)
    "board_size": BOARD_SIZE,  # Tuple (H, W) of the spatial tensor (e.g., (5, 6))
    "action_size": ACTION_SIZE,  # Total size of the policy output vector (e.g., 74)
    "global_feature_size": GLOBAL_FEATURE_SIZE,  # Size of the concatenated global feature vector (e.g., 42)
    "value_head_hidden_dim": 256,  # Size of the hidden layer in the value head's MLP (VALUE_HEAD_HIDDEN_DIM)
    "num_res_blocks": 6,  # Number of residual blocks in the CNN body
    "policy_head_conv_filters": 2,  # Filters in the policy head's initial 1x1 conv
    "value_head_conv_filters": 1,  # Filters in the value head's initial 1x1 conv
}

training_config = {
    # Parameters controlling the training optimization process passed to ModelManager.__init__
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optimizer_type": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,  # L2 regularization strength
    "value_loss_weight": 0.5,
    "policy_loss_weight": 0.5,
    "batch_size": 4,
    # Note: EPOCHS = 1 from original seems to map to NUM_EPOCHS_PER_ITER in self_play_config
}

mcts_config = {
    "num_simulations": 4,  # MCTS simulations per move
    "cpuct": 1.0,  # Exploration constant for PUCT
    # --- Parameters for Dirichlet noise added to root priors during self-play ---
    "dirichlet_alpha": 0.4,
    "dirichlet_epsilon": 0.2,
    # --- Temperature parameter for move selection ---
    "turns_until_tau0": 10,  # Turn after which move selection becomes deterministic (greedy based on visits)
    # Before this turn, visits^(1/tau) is used, tau=1 usually.
    "action_size": model_config["action_size"],
}

eval_mcts_config = {
    "num_simulations": 50,  # MCTS simulations per move
    "cpuct": 1.0,  # Exploration constant for PUCT
    # --- Parameters for Dirichlet noise added to root priors during self-play ---
    "dirichlet_alpha": 0,
    "dirichlet_epsilon": 0,
    # --- Temperature parameter for move selection ---
    "turns_until_tau0": 10,  # Turn after which move selection becomes deterministic (greedy based on visits)
    # Before this turn, visits^(1/tau) is used, tau=1 usually.
    "action_size": model_config["action_size"],
}

self_play_config = {
    "num_iterations": 2,  # Total number of self-play -> train iterations
    "num_games_per_iter": 25,  # Number of games generated per iteration
    "epochs_per_iter": 1,  # Number of training epochs over the buffer per iteration
    "replay_buffer_size": 50000,  # Max number of (s, pi, z) examples stored
    "checkpoint_folder": "./harmonies_az_run/",  # Folder to save model checkpoints (CHECKPOINT_FOLDER)
    # --- Evaluation Settings (run periodically, e.g., after N iterations) ---
    "eval_episodes": 20,  # Number of games to play between current and best model
    "eval_win_rate_threshold": 0.55,  # Win rate needed for new model to become the 'best'
    "eval_frequency": 10,  # How often evaluation is done (every N interations)
    # --- Info needed by helper functions ---
    "action_size": model_config["action_size"],
    "num_hexes": NUM_HEXES,
    "coordinate_to_index_map": coordinate_to_index_map,
}

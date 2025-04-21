from typing import TypedDict, Tuple, Dict, Literal


# For model_config
class ModelConfigType(TypedDict):
    input_channels: int
    cnn_filters: int
    board_size: Tuple[int, int]
    action_size: int
    global_feature_size: int
    value_head_hidden_dim: int
    num_res_blocks: int
    policy_head_conv_filters: int
    value_head_conv_filters: int


# For training_config
class TrainingConfigType(TypedDict):
    device: Literal["cuda", "cpu", "mps"]
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    momentum: float
    value_loss_weight: float
    policy_loss_weight: float
    batch_size: int


# For mcts_config
class MCTSConfigType(TypedDict):
    num_simulations: int
    cpuct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    turns_until_tau0: int
    action_size: int


# For self_play_config
class SelfPlayConfigType(TypedDict):
    num_iterations: int
    num_games_per_iter: int
    num_parallel_games: int
    epochs_per_iter: int
    replay_buffer_size: int
    worker_device: str
    checkpoint_folder: str
    eval_episodes: int
    eval_win_rate_threshold: float
    replay_buffer_folder: str
    replay_buffer_filename: str
    best_model_filename: str
    eval_frequency: int
    action_size: int
    num_hexes: int
    coordinate_to_index_map: Dict[Tuple[int, int], int]

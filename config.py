import torch

#### SELF PLAY
MCTS_SIMS = 50
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8
ACTION_SIZE = 74 # Update this based on your action space
INPUT_CHANNELS = 38 # TODO: Check if this is actually 38
BOARD_SIZE = (5, 6)
NUM_HEXES: 23 # Or 19/etc. based on your final grid
    # Add coordinate_to_index_map (precompute this)
REPLAY_BUFFER_SIZE = 50000
NUM_ITERS = 100 # Total training iterations
NUM_SELF_PLAY_GAMES_PER_ITER = 25 # Games per training iteration
NUM_EPOCHS_PER_ITER =  1 
CHECKPOINT_FOLDER = './harmonies_az_run/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu', 



#### RETRAINING
BATCH_SIZE = 64
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.001
TRAINING_LOOPS = 10
VALUE_LOSS_WEIGHT = .5
POLICY_LOSS_WEIGHT = .5

#### EVALUATION
EVAL_EPISODES = 20

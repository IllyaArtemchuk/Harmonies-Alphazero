import os
import torch.multiprocessing as mp
from model import ModelManager
from trainer import Trainer
from loggers import logger_main
from training_dashboard import create_training_monitor

if __name__ == "__main__":
    os.environ["GAME_DEBUG"] = "0"
    
    # Ensure we're using Connect 4
    from game_config import GAME_TYPE
    assert GAME_TYPE == "connect4", f"Expected GAME_TYPE to be 'connect4', but got '{GAME_TYPE}'"

    logger_main.info("========================================")
    logger_main.info("     CONNECT 4 TRAINING INITIALIZATION  ")
    logger_main.info("========================================")

    # Import Connect 4 specific configurations
    from connect4_config import (
        connect4_model_config,
        connect4_training_config,
        connect4_mcts_config,
        connect4_self_play_config,
    )

    logger_main.info("Using Connect 4 Model Config: %s", connect4_model_config)
    logger_main.info("Using Connect 4 Training Config: %s", connect4_training_config)
    logger_main.info("Using Connect 4 MCTS Config: %s", connect4_mcts_config)
    logger_main.info("Using Connect 4 Self-Play Config: %s", connect4_self_play_config)

    mp.set_start_method("spawn", force=True)

    model_mgr = ModelManager(connect4_model_config, connect4_training_config)

    # Optionally load a checkpoint if continuing a run
    # model_mgr.load_checkpoint(folder=connect4_self_play_config['checkpoint_folder'], filename="iteration_0001.pth.tar")

    # Create experiment name
    experiment_name = "connect4_alphazero"
    
    # Create and run the trainer with metrics enabled
    trainer = Trainer(
        model_mgr,
        connect4_mcts_config,
        connect4_self_play_config,
        connect4_training_config,
        enable_metrics=True,
        experiment_name=experiment_name
    )
    
    # Create training monitor for additional insights
    monitor = create_training_monitor(experiment_name)
    
    logger_main.info("Starting Connect 4 training loop...")
    logger_main.info(f"TensorBoard command: tensorboard --logdir ./runs")
    trainer.run_training_loop()
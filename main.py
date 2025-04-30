import os
import torch.multiprocessing as mp
from model import ModelManager
from trainer import Trainer


if __name__ == "__main__":
    os.environ["GAME_DEBUG"] = "0"
    from config import (
        model_config_default,
        training_config_default,
        mcts_config_default,
        self_play_config_default,
    )

    mp.set_start_method("spawn", force=True)

    model_mgr = ModelManager(model_config_default, training_config_default)

    # Optionally load the very first checkpoint if continuing a run
    # model_mgr.load_checkpoint(folder=self_play_config_default['checkpoint_folder'])

    # Create and run the trainer
    trainer = Trainer(
        model_mgr,
        mcts_config_default,
        self_play_config_default,
        training_config_default,
    )
    trainer.run_training_loop()

from model import ModelManager
from trainer import Trainer

if __name__ == "__main__":
    # Import your configurations
    from config import model_config, training_config, mcts_config, self_play_config

    # Initialize components
    model_mgr = ModelManager(model_config, training_config)

    # Optionally load the very first checkpoint if continuing a run
    # model_mgr.load_checkpoint(folder=self_play_config['checkpoint_folder'])

    # Create and run the trainer
    trainer = Trainer(model_mgr, mcts_config, self_play_config, training_config)
    trainer.run_training_loop()


from config import test_model_config, test_training_config, test_mcts_config, test_self_play_config
from model import ModelManager
from trainer import Trainer

if __name__ == '__main__':
    # Use the TEST configs when initializing
    model_mgr = ModelManager(test_model_config, test_training_config)
    
    # Clear previous test checkpoints/buffer if desired before run
    # import shutil
    # shutil.rmtree(test_self_play_config['checkpoint_folder'], ignore_errors=True)
    # shutil.rmtree(test_self_play_config['replay_buffer_folder'], ignore_errors=True)
    
    trainer = Trainer(model_mgr, test_mcts_config, test_self_play_config, test_training_config)
    trainer.run_training_loop()
    print("Test run completed.")

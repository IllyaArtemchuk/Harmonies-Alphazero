import copy
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import torch
from tqdm import tqdm
from model import ModelManager
from harmonies_engine import HarmoniesGameState
from process_game_state import create_state_tensors
from MCTS import get_best_action_and_pi
from buffer import load_buffer, save_buffer, ReplayBufferDataset
from config_types import (
    TrainingConfigType,
    SelfPlayConfigType,
    MCTSConfigType,
)
from config import mcts_config_eval, test_mcts_config_eval
import loggers as lg


class Trainer:
    def __init__(
        self,
        model_manager,
        mcts_config: MCTSConfigType,
        self_play_config: SelfPlayConfigType,
        training_config: TrainingConfigType,
    ):
        """
        Initializes the AlphaZero Trainer.

        Args:
            model_manager (ModelManager): Instance managing the NN.
            mcts_config (dict): Configuration for MCTS search.
            self_play_config (dict): Configuration for self-play loop.
            training_config (dict): Configuration for NN training.
        """
        lg.logger_main.info("Initializing Trainer...")
        self.model_manager = model_manager
        self.mcts_config = mcts_config
        self.self_play_config = self_play_config
        self.training_config = training_config

        # Initialize or load the replay buffer
        buffer_folder = self.self_play_config["replay_buffer_folder"]
        buffer_file = self.self_play_config["replay_buffer_filename"]
        self.replay_buffer = load_buffer(
            max_size=self_play_config["replay_buffer_size"],
            folder=buffer_folder,
            filename=buffer_file,
        )

        # Evaluation attributes
        self.best_model_manager = (
            None  # Will hold the manager for the best performing model
        )
        self.best_model_filename = (
            "best_model.pth.tar"  # Filename for the best model checkpoint
        )
        self._initialize_best_model()  # Load or initialize the best model

    def execute_self_play_phase(self, data_generating_manager):
        """Runs multiple self-play games in paralell and adds data to the buffer."""
        num_games = self.self_play_config["num_games_per_iter"]
        num_workers = self.self_play_config["num_parallel_games"]
        worker_device = self.self_play_config["worker_device"]
        print(
            f"\n--- Starting Self-Play Phase ({num_games} games using\
            {num_workers} workers on device '{worker_device}') ---"
        )
        start_time = time.time()
        new_examples = 0
        games_completed = 0

        # Currently workers are CPU only, so model is set to cpu
        data_generating_manager.model.cpu()
        model_state_dict = data_generating_manager.model.state_dict()
        # Move model back to its original device if needed
        data_generating_manager.model.to(data_generating_manager.device)

        args_list = [
            (
                copy.deepcopy(model_state_dict),
                copy.deepcopy(
                    self.model_manager.model_config
                ),  # Use candidate's config structure
                copy.deepcopy(
                    self.model_manager.training_config
                ),  # Use candidate's config structure
                copy.deepcopy(self.mcts_config),
                worker_device,
            )
            for _ in range(num_games)
        ]

        # --- Run games in parallel ---
        collected_data = []

        try:
            # Using 'spawn' start method can be more robust on macOS/Windows than 'fork'
            # import torch.multiprocessing as mp
            # mp.set_start_method('spawn', force=True) # Set this early in your main script if needed

            with Pool(processes=num_workers) as pool:
                # Use imap_unordered to get results as they finish, good for progress bars
                # Wrap with tqdm for progress visualization
                results_iterator = pool.imap_unordered(self_play_worker, args_list)

                for game_data in tqdm(
                    results_iterator, total=num_games, desc=" Self-Play Games"
                ):
                    if (
                        game_data
                    ):  # Check if worker returned valid data (not empty list)
                        collected_data.extend(game_data)
                        new_examples += len(game_data)
                        games_completed += 1
                    # else: Game failed in worker, already printed error there

            print("\n  Parallel pool finished.")
        except Exception as e:
            print(f"FATAL ERROR during multiprocessing self-play: {e}")
            # Consider how to handle this - maybe stop training?
            # import traceback; traceback.print_exc()

        # Add collected data to the main replay buffer
        self.replay_buffer.extend(collected_data)

        end_time = time.time()
        print("--- Self-Play Finished ---")
        print(f"  Completed {games_completed}/{num_games} games.")
        print(f"  Added {new_examples} examples.")
        print(f"  Buffer size: {len(self.replay_buffer)} / {self.replay_buffer.maxlen}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

    def execute_training_phase(self):
        """Trains the model using data from the replay buffer."""
        print("\n--- Starting Training Phase ---")
        start_time = time.time()

        if len(self.replay_buffer) < self.training_config["batch_size"]:
            print("  Not enough data in buffer to train yet. Skipping.")
            return

        # Create dataset and dataloader from the current buffer content
        dataset = ReplayBufferDataset(self.replay_buffer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
            num_workers=0,  # Start with 0, increase later if I/O is bottleneck
            pin_memory=True if self.training_config["device"] == "cuda" else False,
        )

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        batches_processed = 0

        for epoch in range(self.self_play_config["epochs_per_iter"]):
            print(
                f"  Training Epoch {epoch+1}/{self.self_play_config['epochs_per_iter']}..."
            )
            for batch in dataloader:
                # Unpack batch - order matches ReplayBufferDataset.__getitem__
                batch_boards, batch_globals, batch_pis, batch_zs = batch

                # Perform one training step
                loss, p_loss, v_loss = self.model_manager.train_step(
                    batch_boards, batch_globals, batch_pis, batch_zs
                )
                total_loss_accum += loss
                policy_loss_accum += p_loss
                value_loss_accum += v_loss
                batches_processed += 1
        lg.logger_main.info(f"  Avg Loss: {total_loss_accum / batches_processed:.4f} (Policy: {policy_loss_accum / batches_processed:.4f},\
                Value: {value_loss_accum / batches_processed:.4f})")
        end_time = time.time()
        print("--- Training Finished ---")
        if batches_processed > 0:
            avg_total_loss = total_loss_accum / batches_processed
            avg_policy_loss = policy_loss_accum / batches_processed
            avg_value_loss = value_loss_accum / batches_processed
            lg.logger_main.info(f"  Avg Loss: {avg_total_loss:.4f} (Policy: {avg_policy_loss:.4f},\
                    Value: {avg_value_loss:.4f})")
            print(
                f"  Avg Loss: {avg_total_loss:.4f} (Policy: {avg_policy_loss:.4f},\
                    Value: {avg_value_loss:.4f})"
            )
            print(f"  Batches processed: {batches_processed}")
        else:
            print("  No batches were processed.")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

    def run_training_loop(self):
        print("============================================")
        print("          STARTING ALPHAZERO TRAINING       ")
        print("============================================")

        num_iterations_config = self.self_play_config["num_iterations"]
        eval_frequency = self.self_play_config["eval_frequency"]

        # Try to load candidate model checkpoint to resume (if any)
        # This also loads optimizer and scheduler state for self.model_manager
        candidate_checkpoint_folder = self.self_play_config["checkpoint_folder"]
        # Use a generic "latest" or "resume" checkpoint name for the candidate
        resume_filename = "latest_candidate.pth.tar"
        loaded_candidate, self.start_iteration = self.model_manager.load_checkpoint(
            folder=candidate_checkpoint_folder, filename=resume_filename
        )
        if loaded_candidate:
            print(f"Resuming training from iteration {self.start_iteration + 1}")
        else:
            print("Starting training from scratch (no candidate checkpoint found or load failed).")
            self.start_iteration = 0


        for iteration_count in range(self.start_iteration, num_iterations_config):
            current_iteration_num = iteration_count + 1 # 1-based for display
            print(f"\n=============== ITERATION {current_iteration_num}/{num_iterations_config} ===============")

            current_lr = self.model_manager.get_current_lr()
            print(f"--- Starting Iteration with LR: {current_lr:.7f} ---")
            lg.logger_main.info(f"Iteration {current_iteration_num} | Current LR: {current_lr:.7f}")

            # Data generation uses the best_model_manager
            data_gen_mgr = self.best_model_manager
            print(f"--- Generating data using best model: {self.best_model_filename} ---")
            self.execute_self_play_phase(data_gen_mgr)

            # Training uses self.model_manager (the candidate)
            self.execute_training_phase()

            # Step the scheduler for self.model_manager (candidate)
            # For ReduceLROnPlateau, you'd pass a metric like validation loss/win_rate here
            # For StepLR, no metric is needed.
            self.model_manager.step_scheduler() # Pass metric if using ReduceLROnPlateau
            new_lr = self.model_manager.get_current_lr()
            if abs(new_lr - current_lr) > 1e-9: # Check if LR actually changed
                 print(f"--- LR updated by scheduler to: {new_lr:.7f} ---")
                 lg.logger_main.info(f"LR updated by scheduler to: {new_lr:.7f}")

            # Save candidate model checkpoint (self.model_manager)
            # This will also save the updated scheduler state
            self.model_manager.save_checkpoint(
                folder=candidate_checkpoint_folder, filename=resume_filename, iteration=current_iteration_num
            )

            # Save replay buffer at the end of each iteration
            save_buffer(
                self.replay_buffer,
                folder=self.self_play_config["replay_buffer_folder"],
                filename=self.self_play_config["replay_buffer_filename"]
            )

            # ... (buffer saving, evaluation logic) ...
            if current_iteration_num % eval_frequency == 0 and current_iteration_num > 0:
                self.evaluate_model() # This updates self.best_model_manager if candidate is better

        print("\n============================================")
        print("             TRAINING COMPLETE             ")
        print("============================================")

    def _initialize_best_model(self):
        """Initializes or loads the 'best' model for comparison."""
        checkpoint_folder = self.self_play_config["checkpoint_folder"]
        best_model_path = Path(checkpoint_folder) / self.best_model_filename

        # Create a separate ModelManager instance for the best model
        best_model_config = copy.deepcopy(self.model_manager.model_config)
        best_training_config = copy.deepcopy(self.model_manager.training_config)

        self.best_model_manager = ModelManager(best_model_config, best_training_config)

        print("\n--- Initializing Best Model ---")
        # Try loading the existing best model checkpoint
        loaded = self.best_model_manager.load_checkpoint(
            folder=checkpoint_folder, filename=self.best_model_filename
        )

        if not loaded:
            print("No existing best model found. Saving current model as initial best.")
            self.model_manager.save_checkpoint(
                folder=checkpoint_folder, filename=self.best_model_filename
            )
            # Reload into best_model_manager to ensure it has the saved state
            self.best_model_manager.load_checkpoint(
                folder=checkpoint_folder, filename=self.best_model_filename
            )
        else:
            print("Loaded existing best model for comparison.")

    def evaluate_model(self):
        """Pits the current model against the best known model."""
        print("\n--- Starting Evaluation Phase ---")
        start_time = time.time()

        num_eval_games = self.self_play_config["eval_episodes"]
        win_threshold = self.self_play_config["eval_win_rate_threshold"]
        checkpoint_folder = self.self_play_config["checkpoint_folder"]

        candidate_wins = 0
        best_wins = 0
        draws = 0

        for i in range(num_eval_games):
            first_player = i % 2  # Alternate starting player (0=candidate, 1=best)
            print(
                f"  Playing evaluation game {i+1}/{num_eval_games}\
                (Candidate plays as P{first_player})..."
            )

            outcome = self.play_one_eval_game(
                self.model_manager, self.best_model_manager, first_player
            )

            if outcome == 1:
                candidate_wins += 1
                print(f"  ...Candidate won.")
            elif outcome == -1:
                best_wins += 1
                print(f"  ...Best model won.")
            else:
                draws += 1
                print(f"  ...Draw or Error.")

        total_non_draws = candidate_wins + best_wins
        if total_non_draws == 0:
            win_rate = 0.5  # Avoid division by zero if all games are draws/errors
        else:
            win_rate = candidate_wins / total_non_draws

        print(f"--- Evaluation Finished ---")
        print(
            f"  Results: Candidate={candidate_wins}, Best={best_wins}, Draws/Errors={draws}"
        )
        print(f"  Candidate Win Rate (vs Best, excluding draws): {win_rate:.3f}")
        lg.logger_main.info(f"--- Evaluation Finished ---")
        lg.logger_main.info( f"  Results: Candidate={candidate_wins}, Best={best_wins}, Draws/Errors={draws}")
        lg.logger_main.info(f"  Candidate Win Rate (vs Best, excluding draws): {win_rate:.3f}")
        # Check if the candidate model is significantly better
        if win_rate > win_threshold: # (candidate_wins / total_non_draws)
                print(f"  Candidate model passed threshold ({win_threshold:.2f})!")
                print(f"  Updating best model checkpoint to '{self.best_model_filename}'...")
                lg.logger_main.info(f"  Candidate model passed threshold ({win_threshold:.2f})!")
                # Save the current candidate model's weights AS the new best model
                self.model_manager.save_checkpoint(
                    folder=self.self_play_config["checkpoint_folder"],
                    filename=self.best_model_filename,
                    iteration=self.start_iteration + (self.self_play_config["num_iterations"] - self.start_iteration) # A bit hacky for iter num here
                )
                # Update the best_model_manager in memory to match
                self.best_model_manager.load_checkpoint( # type: ignore
                    folder=self.self_play_config["checkpoint_folder"], filename=self.best_model_filename
                )
                print("  Best model updated.")
        else:
            print(
                f"  Candidate model did not pass threshold ({win_threshold:.2f}). Best model remains unchanged."
            )
            lg.logger_main.info(f"  Candidate model did not pass threshold ({win_threshold:.2f}). Best model remains unchanged.")
            # Optional: Reload the current model_manager with the best weights if desired
            # self.model_manager.load_checkpoint(folder=checkpoint_folder, filename=self.best_model_filename)

        end_time = time.time()
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

    def play_one_eval_game(self, candidate_manager, best_manager, first_player):
        """
        Plays one game between two models for evaluation.

        Args:
            candidate_manager: ModelManager for the current (candidate) model.
            best_manager: ModelManager for the best known model.
            first_player (int): 0 or 1, indicating which model plays first (0=candidate, 1=best).

        Returns:
            int: Outcome from the perspective of the CANDIDATE model
                 (1 if candidate wins, -1 if best wins, 0 for draw).
        """
        game = HarmoniesGameState()
        players = (
            {0: candidate_manager, 1: best_manager}
            if first_player == 0
            else {0: best_manager, 1: candidate_manager}
        )
        game_move_count = 0 # Initialize for eval game

        while not game.is_game_over():
            current_player_idx = game.get_current_player()
            current_player_manager = players[current_player_idx]
            eval_config = mcts_config_eval
            if self.mcts_config["testing"]:
                eval_config = test_mcts_config_eval
            try:
                # Use a deterministic MCTS search for evaluation (no noise, greedy move selection)
                # We might need a slightly different config or flag in get_best_action_and_pi
                # For now, assume get_best_action_and_pi uses greedy selection when called here
                best_action, _ = get_best_action_and_pi(
                    game.clone(), current_player_manager, eval_config, game_move_count # Pass game_move_count
                )
            except Exception as e:
                print(
                    f"ERROR during MCTS search in EVALUATION game: {e}\nState:\n{game}"
                )
                return 0  # Treat error as a draw or handle differently

            if best_action is None:
                print(
                    f"WARNING: MCTS failed in EVALUATION game for player {current_player_idx}. Treating as draw.\nState:\n{game}"
                )
                return 0

            try:
                game = game.apply_move(best_action)
            except Exception as e:
                print(
                    f"ERROR during apply_move in EVALUATION game: {e}.\nAction: {best_action}"
                )
                return 0  # Treat error as a draw
            game_move_count += 1 # Increment for eval game

        final_outcome = game.get_game_outcome()  # 1 if P0 wins, -1 if P1 wins, 0 Draw
        if final_outcome is None:
            return 0  # Error case

        # Adjust outcome relative to the CANDIDATE model
        if first_player == 0:  # Candidate played as P0
            return final_outcome
        else:  # Candidate played as P1
            return -final_outcome


def self_play_worker(args):
    """
    Runs a single self-play game simulation in a worker process.

    Args:
            model_state_dict (dict): State dictionary of the NN weights.
            model_config (dict): Configuration for the AlphaZeroModel.
            training_config (dict): Training configuration (needed for ModelManager init).
            mcts_config (dict): Configuration for MCTS.
            worker_device (str): 'cpu' or 'cuda'/'mps' - device for this worker's model.
    Returns:
        list: Collected training data [(board_t, global_t, pi_t, z_t)] or empty list on error.
    """
    model_state_dict, model_config, training_config, mcts_config, worker_device = args

    # --- 1. Create local ModelManager and load weights ---
    try:
        # Modify training_config for the worker if needed (e.g., force CPU)
        worker_training_config = training_config.copy()
        worker_training_config["device"] = worker_device

        local_model_manager = ModelManager(model_config, worker_training_config)
        local_model_manager.model.load_state_dict(model_state_dict)
        local_model_manager.model.eval()  # Ensure model is in eval mode
        # print(f"Worker {os.getpid()} created model on {device}") # Debug print
    except Exception as e:
        print(f"WORKER ERROR: Failed to initialize model: {e}")
        return []  # Return empty on failure

    # --- 2. Simulate one game ---
    game = HarmoniesGameState()
    game_history = []
    game_move_count = 0 # Initialize game move counter

    while not game.is_game_over():
        current_player_idx = game.get_current_player()

        try:
            # Use the local model manager for predictions
            state_tensors = create_state_tensors(game)
            state_tensors = tuple(
                item.float() for item in state_tensors
            )  # Ensure Float
            state_representation = state_tensors  # Store the tuple

            best_action, pi_target = get_best_action_and_pi(
                game.clone(), local_model_manager, mcts_config, game_move_count # Pass game_move_count
            )
        except Exception as e:
            print(f"WORKER ERROR: Exception during MCTS: {e}\nState:\n{game}")
            # Consider logging traceback for detailed debugging: import traceback; traceback.print_exc()
            return []

        if best_action is None:
            print(
                f"WORKER WARNING: MCTS failed for player {current_player_idx}. Aborting game.\nState:\n{game}"
            )
            return []

        game_history.append(
            {
                "state_rep": state_representation,
                "player": current_player_idx,
                "pi": pi_target,
            }
        )

        try:
            game = game.apply_move(best_action)
        except Exception as e:
            print(
                f"WORKER ERROR: Exception during apply_move: {e}. Aborting game.\nAction: {best_action}"
            )
            return []
        
        game_move_count += 1 # Increment game_move_count

    final_outcome = game.get_game_outcome()
    if final_outcome is None:
        print("WORKER ERROR: Game ended but outcome is None!")
        return []

    # --- 3. Process Game History ---
    training_data = []
    for history_entry in game_history:
        s_board, s_global = history_entry["state_rep"]
        pi_target_np = history_entry["pi"]
        player_turn = history_entry["player"]

        if final_outcome == 0:
            outcome_perspective = 0.0
        elif player_turn == 0:
            outcome_perspective = float(final_outcome)
        else:
            outcome_perspective = -float(final_outcome)

        # Keep data as tensors for consistency, DataLoader prefers tensors
        training_data.append(
            (
                s_board,
                s_global,
                torch.tensor(pi_target_np, dtype=torch.float),
                torch.tensor([outcome_perspective], dtype=torch.float),
            )
        )

    # print(f"Worker {os.getpid()} finished game with {len(training_data)} examples.") # Debug print
    return training_data

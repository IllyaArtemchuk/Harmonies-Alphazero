import copy
import time
from model import ModelManager
from config import *
from harmonies_engine import HarmoniesGameState
from process_game_state import create_state_tensors
from MCTS import get_best_action_and_pi
from buffer import *


class Trainer:
    def __init__(self, model_manager, mcts_config, self_play_config, training_config):
        """
        Initializes the AlphaZero Trainer.

        Args:
            model_manager (ModelManager): Instance managing the NN.
            mcts_config (dict): Configuration for MCTS search.
            self_play_config (dict): Configuration for self-play loop.
            training_config (dict): Configuration for NN training.
        """
        self.model_manager = model_manager
        self.mcts_config = mcts_config
        self.self_play_config = self_play_config
        self.training_config = training_config

        # Initialize or load the replay buffer
        buffer_folder = self.self_play_config.get("replay_buffer_folder", "buffer")
        buffer_file = self.self_play_config.get(
            "replay_buffer_filename", "replay_buffer.pkl"
        )
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

    def run_self_play_game(self, model_manager):
        """
        Plays one full game using AlphaZero MCTS and collects training examples.

        Args:
            model_manager: The ModelManager instance.
        Returns:
            list: A list of training examples for this game, where each example is a tuple:
                (board_tensor, global_features_tensor, pi_target_tensor, outcome_perspective_tensor).
                Returns empty list if a significant error occurs during the game.
        """
        game = HarmoniesGameState()  # Start new game
        # Store history entries as dictionaries for clarity
        game_history = (
            []
        )  # Stores {'board_rep': tensor, 'global_rep': tensor, 'player': int, 'pi': ndarray}

        while not game.is_game_over():
            current_player_idx = (
                game.get_current_player()
            )  # Get the player index (0 or 1)

            # --- Generate NN Inputs (State Representation) ---
            # Calculate these BEFORE the MCTS search for the current state
            try:
                state_tensors = create_state_tensors(game)
                state_tensors = tuple(
                    item.float() for item in state_tensors
                )  # Ensure Float

            except Exception as e:
                print(f"ERROR: Failed to generate state representation for NN: {e}")
                print(f"State:\n{game}")
                return []  # Abort game if representation fails

            # --- Run MCTS Search ---
            try:
                # Pass the current game state (clone is good practice), model_manager, and configs
                best_action, pi_target = get_best_action_and_pi(
                    game.clone(),  # Pass a clone for safety during search
                    model_manager,
                    self.mcts_config,
                    self.self_play_config,
                )
            except Exception as e:
                print(
                    f"ERROR: Exception during MCTS search (get_best_action_and_pi): {e}"
                )
                print(f"State:\n{game}")
                return []  # Abort game

            # --- Handle MCTS Failure ---
            if best_action is None:
                print(
                    f"WARNING: MCTS failed to return a valid action for player {current_player_idx}. Aborting game."
                )
                print(f"State:\n{game}")
                # Return empty list as this game's data might be unreliable
                return []

            # --- Store History (BEFORE applying move) ---
            # Store the NN inputs, the player, and the MCTS policy result
            game_history.append(
                {
                    "state_rep": state_tensors,  # Store the tuple (board_tensor, global_features)
                    "player": current_player_idx,
                    "pi": pi_target,  # pi_target should be a numpy array from MCTS
                }
            )

            # --- Apply Move ---
            try:
                game = game.apply_move(best_action)
            except Exception as e:
                print(f"ERROR: Exception during game.apply_move: {e}. Aborting game.")
                # Log state *before* the failed move
                # (Access state from the last 'state_representation' or re-generate if needed)
                print(f"Action attempted: {best_action}")
                return []  # Return empty list for this failed game

        # Game over
        final_outcome = game.get_game_outcome()

        if final_outcome is None:  # Should not happen if game_is_over is true
            print("ERROR: Game ended but get_game_outcome returned None!")
            return []

        # --- Process Game History into Final Training Data ---
        training_data = []
        for history_entry in game_history:
            s_board, s_global = history_entry[
                "state_rep"
            ]  # Unpack the stored state representation
            pi_target_np = history_entry["pi"]
            player_turn = history_entry["player"]

            # Determine outcome z from the perspective of player_turn
            if final_outcome == 0:  # Draw
                outcome_perspective = 0.0
            elif player_turn == 0:  # It was Player 0's turn
                outcome_perspective = float(
                    final_outcome
                )  # 1.0 if P0 won, -1.0 if P1 won
            else:  # It was Player 1's turn
                outcome_perspective = -float(
                    final_outcome
                )  # -1.0 if P0 won, 1.0 if P1 won

            # Append final training tuple, converting numpy pi and scalar outcome to tensors
            training_data.append(
                (
                    s_board,  # Already a tensor
                    s_global,  # Already a tensor
                    torch.tensor(
                        pi_target_np, dtype=torch.float
                    ),  # Convert pi numpy array to tensor
                    torch.tensor(
                        [outcome_perspective], dtype=torch.float
                    ),  # Outcome as single-element tensor
                )
            )

        return training_data

    def execute_self_play_phase(self, data_generating_manager):
        """Runs multiple self-play games and adds data to the buffer."""
        num_games = self.self_play_config["num_games_per_iter"]
        print(f"\n--- Starting Self-Play Phase ({num_games} games) ---")
        start_time = time.time()
        new_examples = 0
        games_played = 0

        # TODO: Consider parallelization here using multiprocessing.Pool
        # For simplicity, running sequentially first:
        for i in range(num_games):
            print(f"  Playing game {i+1}/{num_games}...")
            game_data = self.run_self_play_game(data_generating_manager)
            if game_data:  # Only add data if game completed successfully
                self.replay_buffer.extend(game_data)
                new_examples += len(game_data)
                games_played += 1
            else:
                print(f"  Game {i+1} aborted due to error.")

        end_time = time.time()
        print(f"--- Self-Play Finished ---")
        print(f"  Completed {games_played}/{num_games} games.")
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

        end_time = time.time()
        print(f"--- Training Finished ---")
        if batches_processed > 0:
            avg_total_loss = total_loss_accum / batches_processed
            avg_policy_loss = policy_loss_accum / batches_processed
            avg_value_loss = value_loss_accum / batches_processed
            print(
                f"  Avg Loss: {avg_total_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})"
            )
            print(f"  Batches processed: {batches_processed}")
        else:
            print("  No batches were processed.")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

    def run_training_loop(self):
        """Executes the main AlphaZero training loop with evaluation."""
        print("============================================")
        print("          STARTING ALPHAZERO TRAINING       ")
        print("============================================")

        num_iterations = self.self_play_config["num_iterations"]
        eval_frequency = self.self_play_config.get(
            "eval_frequency", 10
        )  # Evaluate every N iterations

        for iteration in range(num_iterations):
            print(
                f"\n=============== ITERATION {iteration+1}/{num_iterations} ==============="
            )

            # Use the BEST model for generating self-play data
            # This is crucial: self-play should always use the strongest agent
            data_generating_manager = self.best_model_manager
            # Make sure Trainer.__init__ loads/initializes best_model_manager correctly

            print(f"--- Generating data using model: {self.best_model_filename} ---")
            # 1. Generate game data using the current *best* model
            # Make sure execute_self_play_phase uses the right model manager
            self.execute_self_play_phase(data_generating_manager)

            # 2. Train the *candidate* model using data from the buffer
            self.execute_training_phase()  # Trains self.model_manager

            # 3. Save *candidate* model checkpoint periodically
            checkpoint_folder = self.self_play_config["checkpoint_folder"]
            candidate_filename = f"iteration_{iteration+1:04d}.pth.tar"
            self.model_manager.save_checkpoint(
                folder=checkpoint_folder, filename=candidate_filename
            )

            # 4. Save replay buffer periodically (optional)
            buffer_folder = self.self_play_config.get("replay_buffer_folder", "buffer")
            if (iteration + 1) % 5 == 0:
                save_buffer(self.replay_buffer, folder=buffer_folder)

            # 5. Evaluation Phase
            if (iteration + 1) % eval_frequency == 0:
                self.evaluate_model()  # Compares self.model_manager vs self.best_model_manager

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

        # Play games, alternating who goes first
        # TODO: Consider parallelization for evaluation games as well
        for i in range(num_eval_games):
            first_player = i % 2  # Alternate starting player (0=candidate, 1=best)
            print(
                f"  Playing evaluation game {i+1}/{num_eval_games} (Candidate plays as P{first_player})..."
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

        # Check if the candidate model is significantly better
        if win_rate > win_threshold:
            print(f"  Candidate model passed threshold ({win_threshold:.2f})!")
            print(f"  Updating best model checkpoint...")
            # Save the current model's weights AS the new best model
            self.model_manager.save_checkpoint(
                folder=checkpoint_folder, filename=self.best_model_filename
            )
            # Update the best_model_manager in memory to match
            self.best_model_manager.load_checkpoint(
                folder=checkpoint_folder, filename=self.best_model_filename
            )
            print("  Best model updated.")
        else:
            print(
                f"  Candidate model did not pass threshold ({win_threshold:.2f}). Best model remains unchanged."
            )
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

        while not game.is_game_over():
            current_player_idx = game.get_current_player()
            current_player_manager = players[current_player_idx]

            try:
                # Use a deterministic MCTS search for evaluation (no noise, greedy move selection)
                # We might need a slightly different config or flag in get_best_action_and_pi
                # For now, assume get_best_action_and_pi uses greedy selection when called here
                best_action, _ = get_best_action_and_pi(
                    game.clone(),
                    current_player_manager,
                    eval_mcts_config,
                    self.self_play_config,  # Use main config, but MCTS should be deterministic
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

        final_outcome = game.get_game_outcome()  # 1 if P0 wins, -1 if P1 wins, 0 Draw
        if final_outcome is None:
            return 0  # Error case

        # Adjust outcome relative to the CANDIDATE model
        if first_player == 0:  # Candidate played as P0
            return final_outcome
        else:  # Candidate played as P1
            return -final_outcome

    # --- Placeholder for Evaluation ---
    # def evaluate_model(self):
    #     print("\n--- Starting Evaluation Phase ---")
    #     # Load best model weights if necessary
    #     # Play N games (e.g., self_play_config['eval_episodes'])
    #     # between self.model_manager and self.best_model_manager
    #     # Track win rate
    #     # If self.model_manager wins > threshold (e.g., 0.55),
    #     # update self.best_model_manager weights and save as 'best_model.pth.tar'
    #     pass


# # ============================================
# # Example Main Script Execution
# # ============================================
# if __name__ == '__main__':
#     # Import your configurations
#     from config import model_config, training_config, mcts_config, self_play_config

#     # Initialize components
#     model_mgr = ModelManager(model_config, training_config)

#     # Optionally load the very first checkpoint if continuing a run
#     # model_mgr.load_checkpoint(folder=self_play_config['checkpoint_folder'])

#     # Create and run the trainer
#     trainer = Trainer(model_mgr, mcts_config, self_play_config, training_config)
#     trainer.run_training_loop()

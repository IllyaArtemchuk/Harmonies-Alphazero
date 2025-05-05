import cProfile
import pstats
import torch
import copy
import numpy as np
import random # Make sure random is imported if used in fallback logic
from config import model_config_default, training_config_default, mcts_config_default, self_play_config_default
from constants import ACTION_SIZE # Example, import others if needed directly by worker logic
from harmonies_engine import HarmoniesGameState
from model import ModelManager, AlphaZeroModel, ResidualBlock # Need model classes for instantiation
from process_game_state import create_state_tensors, get_action_index # Import state/action functions
from MCTS import Node, MCTS, Edge, get_best_action_and_pi # Import MCTS components
import loggers as lg # Import your logger setup

# --- Adapted Worker Function (Runs one game sequentially) ---
# This duplicates the logic from self_play_worker but runs in the main thread for profiling
def run_one_game_for_profiling(model_manager, mcts_config, self_play_config):
    """
    Simulates one game for profiling purposes.
    Uses the provided model_manager directly.
    """
    lg.logger_main.info("--- Starting Single Game Simulation for Profiling ---")
    game = HarmoniesGameState() 
    game_history = [] # We don't need to store history for profiling usually, but keep logic
    game_turn = 0

    while not game.is_game_over():
        current_player_idx = game.get_current_player()
        
        try:
            # Generate NN inputs
            state_tensors = create_state_tensors(game)
            state_tensors = tuple(item.float() for item in state_tensors)
            state_representation = state_tensors 

            # Combine necessary configs for get_best_action_and_pi
            current_config = mcts_config.copy() 
            current_config.update({
                 'action_size': self_play_config['action_size'],
                 'num_hexes': self_play_config['num_hexes'],
                 'coordinate_to_index_map': self_play_config['coordinate_to_index_map'],
                 'eval_mode': False 
            })
            current_config['temperature'] = 1.0 if game_turn < mcts_config['turns_until_tau0'] else 0.0

            # Run MCTS using the *passed* model_manager
            best_action, pi_target = get_best_action_and_pi(
                game.clone(), 
                model_manager, # Use the manager directly
                current_config
            ) 
        except Exception as e:
            lg.logger_main.error(f"PROFILING ERROR: Exception during MCTS: {e}", exc_info=True)
            return [] # Abort on error

        if best_action is None:
            lg.logger_main.warning(f"PROFILING WARNING: MCTS failed for player {current_player_idx}. Aborting game.")
            return [] 
            
        # Store minimal history if needed for debugging post-profile
        game_history.append({'player': current_player_idx, 'pi': pi_target })

        try:
            game = game.apply_move(best_action) 
            game_turn += 0.5
        except Exception as e:
            lg.logger_main.error(f"PROFILING ERROR: Exception during apply_move: {e}. Action: {best_action}", exc_info=True)
            return [] 

    final_outcome = game.get_game_outcome() 
    if final_outcome is None:
         lg.logger_main.error("PROFILING ERROR: Game ended but outcome is None!")
         return []

    lg.logger_main.info(f"--- Single Game Simulation Finished (Outcome: {final_outcome}, Turns: {len(game_history)}) ---")
    # Return something small, like the number of turns, just to confirm completion
    return len(game_history) 

# --- Main Profiling Block ---
if __name__ == "__main__":
    print("Setting up for profiling...")

    # --- Use your DEFAULT configs, but maybe lower simulations ---
    # You want to profile the typical workload, but maybe slightly shorter MCTS
    # to make the profiling run faster. Let's use default sims for now.
    profile_mcts_config = mcts_config_default.copy()
    # profile_mcts_config['num_simulations'] = 20 # Optional: Reduce for faster profile run

    profile_training_config = training_config_default.copy()
    profile_training_config['device'] = 'mps' 

    # Need a model config consistent with training
    profile_model_config = model_config_default.copy()

    try:
        profiling_model_manager = ModelManager(profile_model_config, profile_training_config)
        
        # Optional: Load weights if you want to profile with trained weights
        # loaded = profiling_model_manager.load_checkpoint(
        #     folder=self_play_config_default['checkpoint_folder'], 
        #     filename=self_play_config_default['best_model_filename'] # Or an iteration checkpoint
        # )
        # if not loaded: print("Warning: Profiling with uninitialized model weights.")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize ModelManager for profiling: {e}")
        exit()

    # --- Setup Profiler ---
    profiler = cProfile.Profile()
    
    print("\nStarting profiled game run...")
    
    # Run the function under the profiler
    profiler.enable()
    result = run_one_game_for_profiling(
        profiling_model_manager, 
        profile_mcts_config, 
        self_play_config_default # Pass the full self-play config
    )
    profiler.disable()
    
    print(f"\nProfiled game run finished. Result (e.g., num turns): {result}")

    # --- Analyze and Print Stats ---
    print("\n--- Profiling Results ---")
    # Sort stats by cumulative time spent in function and its sub-calls
    stats = pstats.Stats(profiler).sort_stats('cumulative') 
    
    # Print the top 40 functions by cumulative time
    stats.print_stats(40) 
    
    # Optionally, save full stats to a file for more detailed analysis
    stats_file = "self_play_profile.prof"
    stats.dump_stats(stats_file)
    print(f"\nFull profiling stats saved to {stats_file}")
    print(f"You can visualize this with snakeviz: pip install snakeviz && snakeviz {stats_file}")
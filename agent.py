from MCTS import MCTS
from config import *


class Trainer():
    def __init__(self):
        pass
    # def choose_move_alphazero(self, game_state, model_manager):
    #     """
    #     Runs MCTS simulation to determine the best move from the current state.

    #     Args:
    #         game_state: The current HarmoniesGameState object.
    #         model_manager: Your ModelManager instance containing the NN and predict method.
    #         config: Dictionary with hyperparameters (e.g., MCTS_SIMS, cpuct).

    #     Returns:
    #         tuple: (chosen_move, pi_target)
    #             chosen_move: The action selected by MCTS (e.g., pile index or (tile_idx, coord)).
    #             pi_target: The normalized visit count distribution (training target for policy head).
    #                     Should be a numpy array of size matching the action space (e.g., 74).
    #     """
    #     # 1. Initialize MCTS Tree for this move decision
    #     root_node = Node(game_state) # Assuming Node takes a game_state
    #     mcts = MCTS(root_node, CPUCT) # Assuming MCTS takes root and c_puct

    #     # 2. Run MCTS Simulations
    #     for _ in range(MCTS_SIMS):
    #         # --- Execute one simulation ---
    #         # a. Select leaf node using PUCT (handle moving NN predict call here)
    #         leaf_node, breadcrumbs = mcts.moveToLeaf_and_get_path() # Modify moveToLeaf

    #         # b. Expand & Evaluate Leaf Node (if not terminal)
    #         if not leaf_node.state.is_game_over(): # Check if terminal
    #             # Prepare NN input (spatial + non-spatial)
    #             state_tensor = create_state_tensor(leaf_node.state) # Plus non-spatial features if needed
    #             # Predict policy and value
    #             policy_p, value_v = model_manager.predict(state_tensor)
                
    #             # Expand the node: create children edges/nodes and set prior 'P' using policy_p
    #             mcts.expand_leaf(leaf_node, policy_p) 
    #         else:
    #             # Game is over at the leaf, get the actual outcome
    #             outcome = leaf_node.state.get_game_outcome() # Returns 1, -1, or 0
    #             # Ensure outcome is from the perspective of the player whose turn it *was* at the leaf
    #             # This might need adjustment based on how get_game_outcome works
    #             value_v = outcome # Or adjust perspective if needed 

    #         # c. Backpropagate the value
    #         mcts.backFill(value_v, breadcrumbs) # Pass value and path

    #     # 3. Get Action Probabilities (pi_target) from Root Visit Counts
    #     # This depends on how you store edges/children at the root
    #     root_edges = mcts.get_root_edges() # Need a way to get edges/children of the root
    #     pi_target = np.zeros(config['ACTION_SIZE']) # e.g., 74
    #     total_visits = 0
        
    #     edge_data_for_selection = [] # Store (action, visits, edge) for choosing move

    #     # Iterate through edges/children connected to the root
    #     # The structure depends on your MCTS implementation (dict or list)
    #     for action, edge in root_edges.items(): # Assuming dict: {action: Edge}
    #         action_index = self._get_action_index(action, config) # You need a function to map game action -> policy vector index
    #         pi_target[action_index] = edge.stats['N']
    #         total_visits += edge.stats['N']
    #         edge_data_for_selection.append((action, edge.stats['N']))
            
    #     if total_visits > 0:
    #         pi_target = pi_target / total_visits
    #     else:
    #         # Handle case where no simulations were run or root has no children (shouldn't happen?)
    #         # Maybe assign uniform probability to legal moves? Or raise error?
    #         print("Warning: MCTS root had zero total visits.")
    #         # Fallback: uniform probability over legal moves? Requires care.
    #         legal_moves = game_state.get_legal_moves()
    #         num_legal = len(legal_moves)
    #         if num_legal > 0:
    #             uniform_prob = 1.0 / num_legal
    #             for move in legal_moves:
    #                 action_index = self._get_action_index(move, config)
    #                 pi_target[action_index] = uniform_prob
            

    #     # 4. Choose the Move to Play
    #     # Usually select the move with the highest visit count (greedy)
    #     # Optional: Add temperature parameter for exploration during early self-play
    #     best_action = None
    #     max_visits = -1
    #     for action, visits in edge_data_for_selection:
    #         if visits > max_visits:
    #             max_visits = visits
    #             best_action = action
                
    #     if best_action is None:
    #         # Handle edge case: No valid moves or MCTS failed? Choose randomly?
    #         print("Warning: MCTS could not select a best action. Choosing random legal move.")
    #         legal_moves = game_state.get_legal_moves()
    #         if legal_moves:
    #             best_action = random.choice(legal_moves)
    #         else:
    #             # This means the game should likely have ended.
    #             raise Exception("MCTS failed to find a move and no legal moves exist.")


    #     return best_action, pi_target
    
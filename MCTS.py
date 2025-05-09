import numpy as np
from config import *
from process_game_state import create_state_tensors, get_action_index
import random
import loggers as lg


class Node:
    def __init__(self, state):
        self.state = state
        self.current_player = state.current_player  # Whose turn it is IN THIS STATE
        # Generate a unique ID for the state if it doesn't have one
        # This ID is crucial for the self.tree dictionary lookup in MCTS
        self.id = hash(state)

        self.edges = {}

    def is_leaf(self):
        # A node is a leaf if it has no outgoing edges (hasn't been expanded yet)
        return len(self.edges) == 0


class Edge:
    def __init__(self, in_node, out_node, prior, action):
        # ID might be less critical now if we don't store edges separately
        # self.id = str(inNode.id) + '|' + str(outNode.id)
        self.in_node = in_node
        self.out_node = out_node
        self.current_player = (
            in_node.current_player
        )  # Player who took the action leading to outNode
        self.action = action  # The action taken (e.g., pile_idx or (tile_idx, coord))

        self.stats = {
            "N": 0,  # Visit count
            "W": 0,  # Total action value (sum of values from simulations passing through)
            "Q": 0,  # Mean action value (W/N)
            "P": prior,  # Prior probability from NN policy head
        }


class MCTS:
    def __init__(self, root_node, mcts_config: MCTSConfigType):
        """
        Initializes the MCTS search tree.

        Args:
            root_node (Node): The node representing the starting state of the search.
            cpuct (float): Exploration constant.
        """
        self.root = root_node
        self.tree = {}  # Stores all nodes encountered in this search {node.id: Node}
        self.mcts_config = mcts_config
        self.add_node(root_node)  # Add root node to the tree dictionary

    def __len__(self):
        return len(self.tree)

    def add_node(self, node):
        # Add node to the tree dictionary using its unique ID
        self.tree[node.id] = node

    def move_to_leaf(self):
        """
        Traverses the tree from the root node to a leaf node using the PUCT formula.

        Returns:
            tuple: (leaf_node (Node), breadcrumbs (list[Edge]))
                   leaf_node: The selected leaf node.
                   breadcrumbs: List of edges followed to reach the leaf node.
        """
        lg.logger_mcts.info("------MOVING TO LEAF------")
        breadcrumbs = []
        current_node = self.root

        while not current_node.is_leaf():
            lg.logger_mcts.info(
                "PLAYER TURN at node %s selection: %d",
                current_node.id,
                current_node.current_player,
            )
            
            legal_moves = current_node.state.get_legal_moves()
            if not legal_moves:
                lg.logger_mcts.warning(f"Node {current_node.id} is not a leaf but has no legal moves. Stopping traversal.")
                break # Reached a terminal state effectively

            legal_moves_set = set(legal_moves)
            
            max_qu = -float("inf")
            simulation_edge = None
            simulation_action = None

            # Calculate total visits Ns for the current node's outgoing edges
            ns = 0
            for edge in current_node.edges.values():
                ns += edge.stats["N"]

            sqrt_ns = np.sqrt(max(1.0, ns)) # Avoid sqrt(0)
            
            # Select the edge with the highest PUCT score
            for action, edge in current_node.edges.items():
                if action in legal_moves_set:
                    prior_p = edge.stats['P'] # This P is now potentially noisy if it was the root

                    # PUCT calculation using the (potentially pre-noised) prior_p
                    u = (
                        self.mcts_config["cpuct"]
                        * prior_p 
                        * sqrt_ns
                        / (1 + edge.stats["N"])
                    )

                    q = edge.stats["Q"] 

                    lg.logger_mcts.debug(f"  Action: {action}, Legal: Yes, Q: {q:.3f}, N: {edge.stats['N']}, P(prior): {prior_p:.3f}, U: {u:.3f}, Q+U: {q+u:.3f}")

                    if q + u > max_qu:
                        max_qu = q + u
                        simulation_action = action 
                        simulation_edge = edge     
                else:
                    lg.logger_mcts.debug(f"  Action: {action}, Legal: No, Skipping PUCT.")

            if simulation_edge is None:
                # This can happen if the node has edges, but *none* of them correspond to
                # currently legal moves (e.g., weird game state or bug).
                # Or if legal_moves was empty initially.
                lg.logger_mcts.error(
                    "MCTS Selection failed: Node %s has no legal actions among its existing edges (%d edges total, %d legal moves found). State: %s",
                    current_node.id, len(current_node.edges), len(legal_moves_set), current_node.state
                )
                break # Stop traversal here
                # Handle error: Maybe break? Choose randomly from legal_moves if any exist but had no edges?
                # If legal_moves is not empty, but simulation_edge is None, it implies MCTS hasn't
                # expanded nodes corresponding to those legal moves yet, or there's a mismatch.
                # A robust fallback might be to pick a random legal move and hope MCTS expands it next time.
                # For now, let's break, indicating a likely problem state.

            lg.logger_mcts.info(
                "Selected LEAF action %s with Q+U %.4f", simulation_action, max_qu
            )

            # Move to the next node based on the selected edge
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)

        lg.logger_mcts.info("Reached leaf node %s or stopped traversal.", current_node.id)
        return current_node, breadcrumbs

    def expand_leaf(self, leaf_node, policy_p):
        """
        Expands a leaf node by creating child nodes and edges for all legal moves.
        Initializes the prior probabilities 'P' of the new edges using the NN policy output.

        Args:
            leaf_node (Node): The leaf node to expand.
            policy_p (np.ndarray): Policy vector output from the NN for the leaf node's state.
                                   Should have size ACTION_SIZE.
        """
        lg.logger_mcts.info("------EXPANDING LEAF NODE %s------", leaf_node.id)

        # Get all legal actions from the leaf node's state
        legal_moves = leaf_node.state.get_legal_moves()

        if not legal_moves:
            lg.logger_mcts.warning(
                "Attempting to expand a node with no legal moves (likely terminal)."
            )
            return  # Nothing to expand
        for move in legal_moves:
            # Get the prior probability for this specific move from the NN's policy output
            action_index = get_action_index(move)  # Map game move -> flat index
            prior_p = policy_p[action_index]

            next_state = leaf_node.state.apply_move(move)
            next_state_id = hash(next_state)

            # Ensure the new state has a unique ID
            if not hasattr(next_state, "id") or next_state.id is None:
                # Generate ID if missing (use same method as in Node.__init__)
                next_state.id = hash(next_state)  # Example ID generation

            # Check if the child node already exists in the tree (e.g., transposition)
            if next_state_id in self.tree:
                child_node = self.tree[next_state_id]

                # --- CRITICAL CHECK ---
                if next_state_id == leaf_node.id:
                    lg.logger_mcts.critical(
                        f"CRITICAL LOOP DETECTED: Node {leaf_node.id} expanding move {move} points back to itself!"
                    )
                    # What to do here? Don't add the edge? Raise error?
                    continue  # Avoid adding self-loop edge

                lg.logger_mcts.debug(
                    "Child node %s (state %s) already exists.",
                    child_node.id,
                    next_state.id,
                )
            else:
                # Create a new node for the child state
                child_node = Node(next_state)
                self.add_node(child_node)  # Add the new node to the tree dictionary
                lg.logger_mcts.debug(
                    "Created new child node %s (state %s).",
                    child_node.id,
                    next_state.id,
                )

            # Create the edge connecting the leaf node to the child node
            new_edge = Edge(leaf_node, child_node, prior_p, move)

            # Add the edge to the leaf node's dictionary of outgoing edges
            leaf_node.edges[move] = new_edge
            lg.logger_mcts.debug(
                "Added edge for action %s with prior P=%.4f", move, prior_p
            )

    def back_fill(self, leaf_node, value_v, breadcrumbs):
        """
        Backpropagates the evaluated value ('value_v') up the tree along the path ('breadcrumbs').

        Args:
            leaf_node (Node): The leaf node where the evaluation occurred.
            value_v (float): The value (-1 to 1) obtained from NN evaluation or terminal state.
            breadcrumbs (list[Edge]): The list of edges followed from the root to the leaf.
        """
        lg.logger_mcts.info(
            "------DOING BACKFILL from leaf %s with value %.4f------",
            leaf_node.id,
            value_v,
        )

        # The value 'value_v' is from the perspective of the player whose turn it is at the leaf_node.
        # We need to adjust the sign when updating edges belonging to the *other* player.
        player_at_leaf = leaf_node.current_player

        for edge in reversed(breadcrumbs):  # Go backwards up the path
            # Determine if the value needs to be flipped for this edge's perspective
            # current_player on edge = player who *took the action* leading TO edge.out_node
            if edge.current_player == player_at_leaf:
                direction = (
                    1.0  # Value is from the perspective of the player who made the move
                )
            else:
                direction = -1.0  # Value is from the opponent's perspective

            value_for_edge = value_v * direction

            # Update edge statistics
            edge.stats["N"] += 1
            edge.stats["W"] += value_for_edge
            edge.stats["Q"] = edge.stats["W"] / edge.stats["N"]

            lg.logger_mcts.debug(
                "Updating edge for action %s (player %d): N=%d, W=%.4f, Q=%.4f (value_for_edge=%.4f)",
                edge.action,
                edge.current_player,
                edge.stats["N"],
                edge.stats["W"],
                edge.stats["Q"],
                value_for_edge,
            )

            # edge.in_node.state.render(lg.logger_mcts) # Render the parent node's state

    def get_root_edges(self):
        return self.root.edges


def get_best_action_and_pi(game_state, model_manager, mcts_config: MCTSConfigType, game_move_number: int):
    """
    Runs MCTS simulation to determine the best move from the current state.

    Args:
        game_state: The current HarmoniesGameState object.
        model_manager: Your ModelManager instance containing the NN and predict method.
        mcts_config: Dictionary with hyperparameters (MCTS_SIMS, cpuct, ACTION_SIZE, etc.).
        game_move_number (int): The number of moves already made in the current game (0-indexed).

    Returns:
        tuple: (chosen_move, pi_target)
            chosen_move: The action selected by MCTS.
            pi_target: The normalized visit count distribution (np.ndarray).
    """
    # 1. Initialize MCTS Tree for this specific move decision
    root_node = Node(game_state)
    mcts = MCTS(root_node, mcts_config)
    # 2. Run MCTS Simulations
    for _ in range(mcts_config["num_simulations"]):  # Use MCTS_SIMS from config
        # --- Execute one simulation ---
        # a. Select leaf node using PUCT
        leaf_node, breadcrumbs = mcts.move_to_leaf()

        # b. Expand & Evaluate Leaf Node (if not terminal)
        if not leaf_node.state.is_game_over():
            # Prepare NN input (ensure format matches NN's forward method)
            board_tensor, global_features_tensor = create_state_tensors(leaf_node.state)

            # Predict policy and value using the NN
            policy_p_raw, value_v = model_manager.predict(
                board_tensor, global_features_tensor
            )

            # Apply Dirichlet noise to root's policy priors if it's the root and in training mode
            current_policy_for_expansion = policy_p_raw
            if leaf_node == mcts.root and not mcts_config.get("testing", False):
                legal_moves_root = leaf_node.state.get_legal_moves()
                if legal_moves_root: # Only apply if there are legal moves
                    policy_p_noisy = policy_p_raw.copy() # Work on a copy
                    
                    num_legal_root_moves = len(legal_moves_root)
                    noise_values = np.random.dirichlet(
                        [mcts_config["dirichlet_alpha"]] * num_legal_root_moves
                    )
                    
                    epsilon = mcts_config["dirichlet_epsilon"]
                    
                    for i, move in enumerate(legal_moves_root):
                        action_idx = get_action_index(move) # Map game move -> flat policy index
                        if 0 <= action_idx < len(policy_p_noisy):
                            policy_p_noisy[action_idx] = (1 - epsilon) * policy_p_noisy[action_idx] + epsilon * noise_values[i]
                        else:
                            lg.logger_mcts.warning(f"Dirichlet noise: Action index {action_idx} for move {move} out of bounds for policy vector size {len(policy_p_noisy)}.")
                    current_policy_for_expansion = policy_p_noisy
                    lg.logger_mcts.debug(f"Applied Dirichlet noise to root priors.")
                else:
                    lg.logger_mcts.debug(f"Root node has no legal moves, skipping Dirichlet noise.")

            # Expand the node using the (potentially noisy) policy output
            mcts.expand_leaf(leaf_node, current_policy_for_expansion)
        else:
            # Game is over at the leaf, get the actual outcome
            outcome = leaf_node.state.get_game_outcome()  # Returns 1, -1, or 0
            # Value for backprop is the outcome from the perspective of the player AT THE LEAF NODE
            value_v = (
                float(outcome) if leaf_node.current_player == 0 else -float(outcome)
            )
            if outcome == 0:
                value_v = 0.0  # Handle draw explicitly
            lg.logger_mcts.info(
                "Leaf node %s is terminal. Outcome = %.1f (perspective of player %d)",
                leaf_node.id,
                value_v,
                leaf_node.current_player,
            )

        # c. Backpropagate the obtained value
        mcts.back_fill(
            leaf_node, value_v, breadcrumbs
        )  # Pass leaf_node for perspective check

    # 3. Get Action Probabilities (pi_target) from Root Visit Counts
    root_edges = mcts.get_root_edges()
    pi_target = np.zeros(
        mcts_config["action_size"], dtype=int
    )  # Use ACTION_SIZE from config
    visit_counts = []  # Store (action, visits) for choosing the move
    total_visits = 0

    for action, edge in root_edges.items():
        action_index = get_action_index(action)
        if action_index >= mcts_config["action_size"]:
            lg.logger_mcts.error(
                "Action %s maps to index %d >= ACTION_SIZE %d",
                action,
                action_index,
                mcts_config["action_size"],
            )
            continue

        visits = edge.stats["N"]
        pi_target[action_index] = visits
        visit_counts.append((action, visits))
        total_visits += visits

    if total_visits > 0:
        pi_target = (
            pi_target / total_visits
        )  # Normalize to create probability distribution
    else:
        # Handle case where root was perhaps terminal or no simulations ran
        lg.logger_mcts.warning("MCTS root had zero total visits after simulations.")
        # Fallback: Assign uniform probability over legal moves?
        legal_moves = game_state.get_legal_moves()
        num_legal = len(legal_moves)
        if num_legal > 0:
            uniform_prob = 1.0 / num_legal
            for move in legal_moves:
                action_index = get_action_index(move)
                pi_target[action_index] = uniform_prob

    # 4. Choose the Move to Play
    best_action = None
    max_visits = -1

    # Check if we are in the exploratory phase (training, not testing, and within turn limit)
    is_exploratory_phase = (
        not mcts_config.get("testing", False) and 
        game_move_number < mcts_config["turns_until_tau0"]
    )

    if is_exploratory_phase:
        lg.logger_mcts.info(f"MCTS: Exploratory move selection (Game Move #{game_move_number}, tau=1)")
        if total_visits > 0:
            # Temperature-based sampling (tau=1, so probs are proportional to visit counts)
            actions = [vc[0] for vc in visit_counts]
            probabilities = np.array([vc[1] for vc in visit_counts], dtype=float) / total_visits
            if len(actions) > 0:
                 best_action = actions[np.random.choice(len(actions), p=probabilities)]
            else: # Should not happen if total_visits > 0 and visit_counts populated
                lg.logger_mcts.warning("MCTS: No actions in visit_counts despite total_visits > 0. Falling back.")
                best_action = None # Will trigger fallback logic below
        else: # No visits, should have been handled by pi_target fallback earlier, but as safety
            best_action = None # Will trigger fallback logic below
    else:
        lg.logger_mcts.info(f"MCTS: Greedy move selection (Game Move #{game_move_number} or Testing Mode)")
        # Deterministic: Choose move with highest visit count
        for action, visits in visit_counts:
            if visits > max_visits:
                max_visits = visits
                best_action = action

    if best_action is None:
        lg.logger_mcts.warning(
            "MCTS could not select a best action (exploratory: %s, max_visits/total_visits: %d/%d). Falling back to random from legal moves.",
            is_exploratory_phase,
            max_visits,
            total_visits
        )
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            best_action = random.choice(legal_moves)
        else:
            lg.logger_mcts.error(
                "MCTS failed, and no legal moves exist. Game should have ended."
            )
            return None, pi_target  # Indicate failure

    return best_action, pi_target

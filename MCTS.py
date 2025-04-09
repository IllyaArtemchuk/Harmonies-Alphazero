import numpy as np
from config import * # Assuming this imports MCTS_SIMS, CPUCT, EPSILON, ALPHA, ACTION_SIZE, NUM_HEXES etc.
from process_game_state import create_state_tensors, get_action_index 
import random
import loggers as lg # Your logging setup

# Assuming your HarmoniesGameState class and other necessary imports are available

class Node():
    def __init__(self, state):
        self.state = state
        self.playerTurn = state.playerTurn # Whose turn it is IN THIS STATE
        # Generate a unique ID for the state if it doesn't have one
        # This ID is crucial for the self.tree dictionary lookup in MCTS
        if not hasattr(state, 'id') or state.id is None:
             # Simple hash example - replace with a robust unique ID generator if needed
             self.id = hash(str(state)) # Example: Hash the string representation
             state.id = self.id # Assign back to state if needed by MCTS tree lookup
        else:
             self.id = state.id
             
        self.edges = {} # Changed from list to dictionary {action: Edge}

    def isLeaf(self):
        # A node is a leaf if it has no outgoing edges (hasn't been expanded yet)
        return len(self.edges) == 0

class Edge():
    def __init__(self, inNode, outNode, prior, action):
        # ID might be less critical now if we don't store edges separately
        # self.id = str(inNode.id) + '|' + str(outNode.id) 
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.playerTurn # Player who took the action leading to outNode
        self.action = action # The action taken (e.g., pile_idx or (tile_idx, coord))

        self.stats =  {
                    'N': 0,  # Visit count
                    'W': 0,  # Total action value (sum of values from simulations passing through)
                    'Q': 0,  # Mean action value (W/N)
                    'P': prior, # Prior probability from NN policy head
                }

class MCTS():
    def __init__(self, root_node, mcts_config):
        """
        Initializes the MCTS search tree.

        Args:
            root_node (Node): The node representing the starting state of the search.
            cpuct (float): Exploration constant.
        """
        self.root = root_node
        self.tree = {} # Stores all nodes encountered in this search {node.id: Node}
        self.cpuct = mcts_config['cpuct']
        self.addNode(root_node) # Add root node to the tree dictionary

    def __len__(self):
        return len(self.tree)

    def addNode(self, node):
        # Add node to the tree dictionary using its unique ID
        self.tree[node.id] = node

    def moveToLeaf(self):
        """
        Traverses the tree from the root node to a leaf node using the PUCT formula.

        Returns:
            tuple: (leaf_node (Node), breadcrumbs (list[Edge]))
                   leaf_node: The selected leaf node.
                   breadcrumbs: List of edges followed to reach the leaf node.
        """
        lg.logger_mcts.info('------MOVING TO LEAF------')
        breadcrumbs = []
        currentNode = self.root

        while not currentNode.isLeaf():
            lg.logger_mcts.info('PLAYER TURN at node %s selection: %d', currentNode.id, currentNode.playerTurn)
            maxQU = -float('inf')
            simulationEdge = None
            simulationAction = None

            # Add Dirichlet noise at the root for exploration
            if currentNode == self.root:
                epsilon = mcts_config['dirichlet_epsilon']
                nu = np.random.dirichlet(mcts_config['dirichlet_alpha'] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges) # Placeholder, only used if epsilon > 0

            # Calculate total visits Ns for the current node's outgoing edges
            Ns = 0
            for edge in currentNode.edges.values(): # Iterate through Edge objects in dict
                Ns += edge.stats['N']

            # Select the edge with the highest PUCT score
            # Use items() to get action and edge, enumerate for nu index
            for idx, (action, edge) in enumerate(currentNode.edges.items()):
                # PUCT calculation
                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Ns) / (1 + edge.stats['N'])
                Q = edge.stats['Q']

                # --- Optional: Log PUCT details ---
                # lg.logger_mcts.info(...) 

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action # Store the action itself
                    simulationEdge = edge   # Store the edge object

            if simulationEdge is None:
                 # This should not happen if a node is not a leaf (must have edges)
                 # unless maybe all priors P were zero? Handle robustly.
                 lg.logger_mcts.error("MCTS Selection failed: Node %s is not leaf but no edge selected.", currentNode.id)
                 # Handle error: maybe break, maybe choose randomly? For now, let's raise.
                 raise Exception(f"MCTS Selection Failure at Node {currentNode.id}")


            lg.logger_mcts.info('Selected action %s with Q+U %.4f', simulationAction, maxQU)

            # Move to the next node based on the selected edge
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge) # Add the edge taken to the path

        lg.logger_mcts.info('Reached leaf node %s', currentNode.id)
        # Return the leaf node found and the path taken
        return currentNode, breadcrumbs

    def expand_leaf(self, leaf_node, policy_p):
        """
        Expands a leaf node by creating child nodes and edges for all legal moves.
        Initializes the prior probabilities 'P' of the new edges using the NN policy output.

        Args:
            leaf_node (Node): The leaf node to expand.
            policy_p (np.ndarray): Policy vector output from the NN for the leaf node's state.
                                   Should have size ACTION_SIZE.
        """
        lg.logger_mcts.info('------EXPANDING LEAF NODE %s------', leaf_node.id)
        # Get all legal actions from the leaf node's state
        legal_moves = leaf_node.state.get_legal_moves()

        if not legal_moves:
            lg.logger_mcts.warning("Attempting to expand a node with no legal moves (likely terminal).")
            return # Nothing to expand

        for move in legal_moves:
            # Get the prior probability for this specific move from the NN's policy output
            action_index = get_action_index(move) # Map game move -> flat index
            prior_p = policy_p[action_index]

            # Create the next state by applying the move (MUST return a NEW state object)
            next_state = leaf_node.state.apply_move(move)
            # Ensure the new state has a unique ID
            if not hasattr(next_state, 'id') or next_state.id is None:
                 # Generate ID if missing (use same method as in Node.__init__)
                 next_state.id = hash(str(next_state)) # Example ID generation
            
            # Check if the child node already exists in the tree (e.g., transposition)
            if next_state.id in self.tree:
                child_node = self.tree[next_state.id]
                lg.logger_mcts.debug('Child node %s (state %s) already exists.', child_node.id, next_state.id)
            else:
                # Create a new node for the child state
                child_node = Node(next_state)
                self.addNode(child_node) # Add the new node to the tree dictionary
                lg.logger_mcts.debug('Created new child node %s (state %s).', child_node.id, next_state.id)

            # Create the edge connecting the leaf node to the child node
            new_edge = Edge(leaf_node, child_node, prior_p, move)

            # Add the edge to the leaf node's dictionary of outgoing edges
            leaf_node.edges[move] = new_edge
            lg.logger_mcts.debug('Added edge for action %s with prior P=%.4f', move, prior_p)

    def backFill(self, leaf_node, value_v, breadcrumbs):
        """
        Backpropagates the evaluated value ('value_v') up the tree along the path ('breadcrumbs').

        Args:
            leaf_node (Node): The leaf node where the evaluation occurred.
            value_v (float): The value (-1 to 1) obtained from NN evaluation or terminal state.
            breadcrumbs (list[Edge]): The list of edges followed from the root to the leaf.
        """
        lg.logger_mcts.info('------DOING BACKFILL from leaf %s with value %.4f------', leaf_node.id, value_v)

        # The value 'value_v' is from the perspective of the player whose turn it is at the leaf_node.
        # We need to adjust the sign when updating edges belonging to the *other* player.
        player_at_leaf = leaf_node.playerTurn

        for edge in reversed(breadcrumbs): # Go backwards up the path
            # Determine if the value needs to be flipped for this edge's perspective
            # playerTurn on edge = player who *took the action* leading TO edge.outNode
            if edge.playerTurn == player_at_leaf:
                direction = 1.0 # Value is from the perspective of the player who made the move
            else:
                direction = -1.0 # Value is from the opponent's perspective

            value_for_edge = value_v * direction

            # Update edge statistics
            edge.stats['N'] += 1
            edge.stats['W'] += value_for_edge
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            lg.logger_mcts.debug('Updating edge for action %s (player %d): N=%d, W=%.4f, Q=%.4f (value_for_edge=%.4f)'
                , edge.action, edge.playerTurn, edge.stats['N'], edge.stats['W'], edge.stats['Q'], value_for_edge)

            # Optional: Render state for debugging if needed
            # edge.inNode.state.render(lg.logger_mcts) # Render the parent node's state


    def get_root_edges(self):
        return self.root.edges


def get_best_action_and_pi(game_state, model_manager, mcts_config):
    """
    Runs MCTS simulation to determine the best move from the current state.

    Args:
        game_state: The current HarmoniesGameState object.
        model_manager: Your ModelManager instance containing the NN and predict method.
        config: Dictionary with hyperparameters (MCTS_SIMS, cpuct, ACTION_SIZE, etc.).

    Returns:
        tuple: (chosen_move, pi_target)
            chosen_move: The action selected by MCTS.
            pi_target: The normalized visit count distribution (np.ndarray).
    """
    # 1. Initialize MCTS Tree for this specific move decision
    root_node = Node(game_state)
    mcts = MCTS(root_node, mcts_config['cpuct']) # Use cpuct from config

    # 2. Run MCTS Simulations
    for _ in range(mcts_config['num_simulations']): # Use MCTS_SIMS from config
        # --- Execute one simulation ---
        # a. Select leaf node using PUCT
        leaf_node, breadcrumbs = mcts.moveToLeaf()

        # b. Expand & Evaluate Leaf Node (if not terminal)
        if not leaf_node.state.is_game_over():
            # Prepare NN input (ensure format matches NN's forward method)
            board_tensor, global_features_tensor = create_state_tensors(leaf_node.state)

            # Predict policy and value using the NN
            policy_p, value_v = model_manager.predict(board_tensor, global_features_tensor)

            # Expand the node using the NN's policy output
            mcts.expand_leaf(leaf_node, policy_p)
        else:
            # Game is over at the leaf, get the actual outcome
            outcome = leaf_node.state.get_game_outcome() # Returns 1, -1, or 0
            # Value for backprop is the outcome from the perspective of the player AT THE LEAF NODE
            value_v = float(outcome) if leaf_node.playerTurn == 0 else -float(outcome)
            if outcome == 0: value_v = 0.0 # Handle draw explicitly
            lg.logger_mcts.info("Leaf node %s is terminal. Outcome = %.1f (perspective of player %d)",
                                leaf_node.id, value_v, leaf_node.playerTurn)

        # c. Backpropagate the obtained value
        mcts.backFill(leaf_node, value_v, breadcrumbs) # Pass leaf_node for perspective check

    # 3. Get Action Probabilities (pi_target) from Root Visit Counts
    root_edges = mcts.get_root_edges()
    pi_target = np.zeros(mcts_config['action_size']) # Use ACTION_SIZE from config
    visit_counts = [] # Store (action, visits) for choosing the move
    total_visits = 0

    for action, edge in root_edges.items():
        action_index = get_action_index(action)
        if action_index >= mcts_config['action_size']:
             lg.logger_mcts.error("Action %s maps to index %d >= ACTION_SIZE %d", action, action_index, mcts_config['action_size'])
             continue 
        
        visits = edge.stats['N']
        pi_target[action_index] = visits
        visit_counts.append((action, visits))
        total_visits += visits

    if total_visits > 0:
        pi_target = pi_target / total_visits # Normalize to create probability distribution
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
    # Deterministic: Choose move with highest visit count
    # TODO: Implement temperature parameter for exploration if needed, especially early in training
    best_action = None
    max_visits = -1
    for action, visits in visit_counts:
        if visits > max_visits:
            max_visits = visits
            best_action = action

    if best_action is None:
        lg.logger_mcts.warning("MCTS could not select a best action (max_visits = %d). Falling back to random.", max_visits)
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            best_action = random.choice(legal_moves)
        else:
            lg.logger_mcts.error("MCTS failed, and no legal moves exist. Game should have ended.")
            # This state suggests an issue earlier in the game logic or MCTS.
            # Depending on robustness needs, you might return None or raise an Exception.
            # For now, let's return None and let the caller handle it.
            return None, pi_target # Indicate failure

    return best_action, pi_target
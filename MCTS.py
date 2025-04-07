import numpy as np
from config import *
from funcs import create_state_tensor, get_action_index
import random

import loggers as lg

class Node():

	def __init__(self, state):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.playerTurn
		self.action = action

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}
				

class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	def moveToLeaf(self):

		lg.logger_mcts.info('------MOVING TO LEAF------')

		breadcrumbs = []
		currentNode = self.root

		value = 0

		while not currentNode.isLeaf():

			lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			maxQU = -99999

			if currentNode == self.root:
				# randomness to encourage exploration at the root node
				epsilon = EPSILON
				nu = np.random.dirichlet([ALPHA] * len(currentNode.edges))
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			Ns = 0 # Total visit count for the state, calculated by adding up the visit counts of the edges
			for action, edge in currentNode.edges:
				Ns = Ns + edge.stats['N']

			for idx, (action, edge) in enumerate(currentNode.edges):

				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Ns) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']

				lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
					, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
					, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

				if Q + U > maxQU:
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge

			lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)


		return currentNode, value, breadcrumbs



	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1

			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value * direction
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
				, value * direction
				, playerTurn
				, edge.stats['N']
				, edge.stats['W']
				, edge.stats['Q']
				)

			edge.outNode.state.render(lg.logger_mcts)


	def get_best_action_and_pi(self, game_state, model_manager):
		"""
		Runs MCTS simulation to determine the best move from the current state.

		Args:
			game_state: The current HarmoniesGameState object.
			model_manager: Your ModelManager instance containing the NN and predict method.
			config: Dictionary with hyperparameters (e.g., MCTS_SIMS, cpuct).

		Returns:
			tuple: (chosen_move, pi_target)
				chosen_move: The action selected by MCTS (e.g., pile index or (tile_idx, coord)).
				pi_target: The normalized visit count distribution (training target for policy head).
						Should be a numpy array of size matching the action space (e.g., 74).
		"""
		# 1. Initialize MCTS Tree for this move decision
		root_node = Node(game_state) # Assuming Node takes a game_state


		# 2. Run MCTS Simulations
		for _ in range(MCTS_SIMS):
			# --- Execute one simulation ---
			# a. Select leaf node using PUCT (handle moving NN predict call here)
			leaf_node, value, breadcrumbs = self.moveToLeaf() # Modify moveToLeaf

			# b. Expand & Evaluate Leaf Node (if not terminal)
			if not leaf_node.state.is_game_over(): # Check if terminal
				# Prepare NN input (spatial + non-spatial)
				state_tensor = create_state_tensor(leaf_node.state) # Plus non-spatial features if needed
				# Predict policy and value
				policy_p, value_v = model_manager.predict(state_tensor)
				
				# Expand the node: create children edges/nodes and set prior 'P' using policy_p
				self.expand_leaf(leaf_node, policy_p) 
			else:
				# Game is over at the leaf, get the actual outcome
				outcome = leaf_node.state.get_game_outcome() # Returns 1, -1, or 0
				# Ensure outcome is from the perspective of the player whose turn it *was* at the leaf
				# This might need adjustment based on how get_game_outcome works
				value_v = outcome # Or adjust perspective if needed 

			# c. Backpropagate the value
			self.backFill(value_v, breadcrumbs) # Pass value and path

		# 3. Get Action Probabilities (pi_target) from Root Visit Counts
		# This depends on how you store edges/children at the root
		root_edges = self.get_root_edges() # Need a way to get edges/children of the root
		pi_target = np.zeros(ACTION_SIZE)
		total_visits = 0
		
		edge_data_for_selection = [] # Store (action, visits, edge) for choosing move

		# Iterate through edges/children connected to the root
		# The structure depends on your MCTS implementation (dict or list)
		for action, edge in root_edges.items(): # Assuming dict: {action: Edge}
			action_index = get_action_index(action) # You need a function to map game action -> policy vector index
			pi_target[action_index] = edge.stats['N']
			total_visits += edge.stats['N']
			edge_data_for_selection.append((action, edge.stats['N']))
			
		if total_visits > 0:
			pi_target = pi_target / total_visits
		else:
			# Handle case where no simulations were run or root has no children (shouldn't happen?)
			# Maybe assign uniform probability to legal moves? Or raise error?
			print("Warning: MCTS root had zero total visits.")
			# Fallback: uniform probability over legal moves? Requires care.
			legal_moves = game_state.get_legal_moves()
			num_legal = len(legal_moves)
			if num_legal > 0:
				uniform_prob = 1.0 / num_legal
				for move in legal_moves:
					action_index = get_action_index(move)
					pi_target[action_index] = uniform_prob
			

		# 4. Choose the Move to Play
		# Usually select the move with the highest visit count (greedy)
		# Optional: Add temperature parameter for exploration during early self-play
		best_action = None
		max_visits = -1
		for action, visits in edge_data_for_selection:
			if visits > max_visits:
				max_visits = visits
				best_action = action
				
		if best_action is None:
			# Handle edge case: No valid moves or MCTS failed? Choose randomly?
			print("Warning: MCTS could not select a best action. Choosing random legal move.")
			legal_moves = game_state.get_legal_moves()
			if legal_moves:
				best_action = random.choice(legal_moves)
			else:
				# This means the game should likely have ended.
				raise Exception("MCTS failed to find a move and no legal moves exist.")


		return best_action, pi_target

	def addNode(self, node):
		self.tree[node.id] = node

	def get_root_edges(self):
		return self.root.edges
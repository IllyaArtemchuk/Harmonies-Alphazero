import random
import copy
import numpy as np
from constants import *
from collections import deque  # For BFS

# --- Verification ---
expected_count = 23
if len(VALID_HEXES) != expected_count:
    raise ValueError(
        f"FATAL: Generated hex grid has {len(VALID_HEXES)} hexes, expected {expected_count}. Check logic."
    )
PLAYER_BOARD_HEX_COUNT = expected_count


# --- Water Scoring Table (unchanged) ---
WATER_SCORES = {1: 0, 2: 2, 3: 5, 4: 8, 5: 11, 6: 15}


def get_water_score(length):
    if length <= 0:
        return 0
    if length in WATER_SCORES:
        return WATER_SCORES[length]
    else:
        return WATER_SCORES[6] + (length - 6) * 4


# --- Helper Functions (get_neighbors, bfs_shortest_path - unchanged logic, uses new VALID_HEXES) ---
def get_neighbors(coord):
    if coord not in VALID_HEXES:
        return []
    q, r = coord
    neighbors = set()
    for dq, dr in AXIAL_DIRECTIONS:
        nq, nr = q + dq, r + dr
        if (
            nq,
            nr,
        ) in VALID_HEXES:  # Check if the potential neighbor is in our defined grid
            neighbors.add((nq, nr))
    return list(neighbors)


def bfs_shortest_path(start_node, end_node, graph_nodes, get_adj_func):
    # (Implementation is identical to previous versions)
    if start_node == end_node:
        return 0
    queue = deque([(start_node, 0)])
    visited = {start_node}
    while queue:
        current_node, distance = queue.popleft()
        neighbors = get_adj_func(current_node)
        for neighbor in neighbors:
            if neighbor == end_node:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return float("inf")


# --- GameState Class ---
class HarmoniesGameState:
    # --- __init__ (unchanged) ---
    def __init__(self, initial_state=None):
        if initial_state:
            self.__dict__.update(initial_state)
        else:
            self.player_boards = [{}, {}]
            self.tile_bag = INITIAL_BAG.copy()
            self.available_piles = []
            self.current_player = 0
            self.tiles_in_hand = []
            self.turn_phase = "choose_pile"
            self.game_over = False
            self.winner = None
            self.final_scores = [0, 0]
            self._replenish_piles()

    # --- Core Methods (_draw_tiles, _replenish_piles, etc.) ---
    # --- These remain unchanged as their logic doesn't depend ---
    # --- directly on the specific grid shape, only on the ---
    # --- PLAYER_BOARD_HEX_COUNT and the VALID_HEXES set      ---
    # --- used by helper functions like get_legal_moves.      ---

    def _draw_tiles(self, num_tiles):
        drawn = []
        flat_bag = [t for t, c in self.tile_bag.items() for _ in range(c)]
        if not flat_bag:
            return []
        actual_draw_count = min(num_tiles, len(flat_bag))
        drawn_indices = random.sample(range(len(flat_bag)), actual_draw_count)
        drawn = [flat_bag[i] for i in drawn_indices]
        for tile_type in drawn:
            self.tile_bag[tile_type] -= 1
        return drawn

    def _replenish_piles(self):
        while len(self.available_piles) < NUM_PILES:
            pile = self._draw_tiles(PILE_SIZE)
            if not pile:
                break
            self.available_piles.append(pile)

    def get_current_player(self):
        return self.current_player

    def _get_top_tile(self, board, coord):
        return board.get(coord, [None])[-1]

    # --- get_legal_moves (iterates over the NEW VALID_HEXES set) ---
    def get_legal_moves(self):
        legal_moves = []
        player = self.current_player
        board = self.player_boards[player]

        if self.turn_phase == "choose_pile":
            # Moves are just pile indices
            return list(range(len(self.available_piles)))

        elif self.turn_phase.startswith("place_tile"):
            if not self.tiles_in_hand:
                return []  # Should not happen if logic is correct

            # Iterate through each tile currently in hand
            for tile_index, tile_to_place in enumerate(self.tiles_in_hand):
                # For this specific tile, find all legal coordinates
                for coord in VALID_HEXES:
                    is_legal_placement = False
                    if coord not in board:  # Placing on an empty spot is always allowed
                        is_legal_placement = True
                    else:
                        # Check stacking rules for THIS tile_to_place
                        stack = board[coord]
                        top_tile = stack[-1]
                        height = len(stack)

                        if tile_to_place == PLANT and top_tile == WOOD and height <= 2:
                            is_legal_placement = True
                        elif (
                            tile_to_place == STONE and top_tile == STONE and height < 3
                        ):
                            is_legal_placement = True
                        elif (
                            tile_to_place == BUILDING
                            and top_tile in [WOOD, STONE, BUILDING]
                            and height < 2
                        ):
                            is_legal_placement = True
                        # No other stacking allowed by default

                    # If placing this tile at this coord is legal, add the move
                    if is_legal_placement:
                        # The move specifies WHICH tile and WHERE
                        legal_moves.append((tile_index, coord))

            return legal_moves  # Returns list of (tile_idx, (q,r)) tuples
        else:
            # Should not happen in valid phases
            return []

    # --- apply_move (validates coords against NEW VALID_HEXES) ---
    def apply_move(self, move):
        new_state = self.clone()
        player = new_state.current_player
        board = new_state.player_boards[player]

        if new_state.turn_phase == "choose_pile":
            # --- Pile Choice Logic (remains the same) ---
            pile_index = move
            if not isinstance(pile_index, int) or not (
                0 <= pile_index < len(new_state.available_piles)
            ):
                raise ValueError(f"Invalid pile index: {pile_index}")
            new_state.tiles_in_hand = new_state.available_piles.pop(pile_index)
            # When pile is chosen, tiles enter hand. Placement order is chosen turn-by-turn.
            new_state.turn_phase = "place_tile_1"
            # --- End Pile Choice Logic ---

        elif new_state.turn_phase.startswith("place_tile"):
            # --- Tile Placement Logic (Modified) ---
            # Expect move to be (tile_index, coord)
            if not (
                isinstance(move, tuple)
                and len(move) == 2
                and isinstance(move[0], int)
                and isinstance(move[1], tuple)
            ):
                raise ValueError(
                    f"Invalid move format for placement phase: {move}. Expected (tile_index, (q, r))"
                )

            tile_index, coord = move

            # Validate tile_index
            if not (0 <= tile_index < len(new_state.tiles_in_hand)):
                raise ValueError(
                    f"Invalid tile_index {tile_index} for hand {new_state.tiles_in_hand}"
                )
            # Validate coordinate
            if coord not in VALID_HEXES:
                raise ValueError(f"Invalid coordinate: {coord}")

            # Get the specific tile from hand using index AND remove it
            tile_to_place = new_state.tiles_in_hand.pop(tile_index)

            # Perform placement and legality checks (using the chosen tile)
            is_legal = False  # Re-check legality for safety
            if coord not in board:
                is_legal = True
                board[coord] = [tile_to_place]  # Place as new stack
            else:
                stack = board[coord]
                top = stack[-1]
                h = len(stack)
                if tile_to_place == PLANT and top == WOOD and h <= 2:
                    is_legal = True
                elif tile_to_place == STONE and top == STONE and h < 3:
                    is_legal = True
                elif (
                    tile_to_place == BUILDING
                    and top in [WOOD, STONE, BUILDING]
                    and h < 2
                ):
                    is_legal = True

                if is_legal:
                    board[coord].append(tile_to_place)  # Add to existing stack
                else:
                    # This should ideally not happen if get_legal_moves is correct
                    # Need to put the tile back in hand before raising error for consistent state
                    new_state.tiles_in_hand.insert(tile_index, tile_to_place)
                    raise ValueError(
                        f"Illegal move attempted in apply_move: Cannot place {tile_to_place} on {coord} with {stack}"
                    )

            # Advance turn phase (logic remains the same)
            if new_state.turn_phase == "place_tile_1":
                new_state.turn_phase = "place_tile_2"
            elif new_state.turn_phase == "place_tile_2":
                new_state.turn_phase = "place_tile_3"
            elif new_state.turn_phase == "place_tile_3":
                new_state._end_turn_actions()
            # --- End Tile Placement Logic ---

        else:
            raise ValueError(f"Invalid turn phase: {new_state.turn_phase}")

        return new_state

    # --- _end_turn_actions (uses PLAYER_BOARD_HEX_COUNT = 23) ---
    def _end_turn_actions(self):
        player_finished = self.current_player
        board = self.player_boards[player_finished]
        empty_hexes = PLAYER_BOARD_HEX_COUNT - len(board)  # Use 23
        player_triggered_end = empty_hexes <= EMPTY_HEX_END_THRESHOLD

        bag_empty_before = sum(self.tile_bag.values()) == 0
        self._replenish_piles()
        bag_empty_trigger = bag_empty_before and not self.available_piles

        end_triggered = player_triggered_end or bag_empty_trigger
        currently_ending = self.game_over

        if end_triggered and not currently_ending:
            self.game_over = True
            if player_finished == 0:  # P1 triggers, P2 gets turn
                self.current_player = 1
                self.turn_phase = "choose_pile"
            else:  # P2 triggers or bag empty, end now
                self.turn_phase = "game_over"
                self._calculate_final_scores()
                self._determine_winner()
        elif currently_ending:  # P2 finishes last turn
            self.turn_phase = "game_over"
            self._calculate_final_scores()
            self._determine_winner()
        else:  # Standard turn switch
            self.current_player = 1 - self.current_player
            self.turn_phase = "choose_pile"

    # --- Game End/Outcome Methods (unchanged) ---
    def is_game_over(self):
        return self.game_over and self.winner is not None

    def get_game_outcome(self):
        if not self.is_game_over():
            return None
        if self.winner == 0:
            return 1
        if self.winner == 1:
            return -1
        return 0

    def _calculate_final_scores(self):
        self.final_scores[0] = self._calculate_score_for_player(0)
        self.final_scores[1] = self._calculate_score_for_player(1)

    def _determine_winner(self):
        if self.final_scores[0] > self.final_scores[1]:
            self.winner = 0
        elif self.final_scores[1] > self.final_scores[0]:
            self.winner = 1
        else:
            self.winner = -1

    # --- Scoring Methods (unchanged logic, rely on get_neighbors for new grid) ---
    def _calculate_score_for_player(self, player_id):
        board = self.player_boards[player_id]
        score = 0
        score += self._score_grass(board)
        score += self._score_mountains(board)
        score += self._score_fields(board)
        score += self._score_buildings(board)
        score += self._score_water(board)
        return score

    def _score_grass(self, board):
        score = 0
        for coord, stack in board.items():
            if not stack:  # If stack is empty, skip to the next item
                continue

            # Only assign top and h if stack is NOT empty
            top = stack[-1]
            h = len(stack)

            if top == PLANT:
                if h == 1:
                    score += 1
                elif h == 2 and stack[0] == WOOD:
                    score += 3
                elif h == 3 and stack[0] == WOOD and stack[1] == WOOD:
                    score += 7
        print(
            "player "
            + str(self.current_player)
            + "scored "
            + str(score)
            + " points with grass!"
        )
        return score

    def _score_mountains(self, board):
        score = 0
        for coord, stack in board.items():
            if not stack:  # If stack is empty, skip
                continue

            # Only assign top and h if stack is NOT empty
            top = stack[-1]
            h = len(stack)

            if top == STONE:
                is_adj = any(
                    self._get_top_tile(board, nc) == STONE
                    for nc in get_neighbors(coord)
                )
                if is_adj:
                    if h == 1:
                        score += 1
                    elif h == 2:
                        score += 3
                    elif h == 3:
                        score += 7
        print(
            "player "
            + str(self.current_player)
            + "scored "
            + str(score)
            + " points with mountains!"
        )
        return score

    def _score_fields(self, board):
        score = 0
        visited = set()
        fields = [c for c, s in board.items() if self._get_top_tile(board, c) == FIELD]
        for start in fields:
            if start in visited:
                continue
            comp = set()
            q = deque([start])
            visited.add(start)
            comp.add(start)
            while q:
                curr = q.popleft()
                for n in get_neighbors(curr):
                    if n not in visited and self._get_top_tile(board, n) == FIELD:
                        visited.add(n)
                        comp.add(n)
                        q.append(n)
            if len(comp) >= 2:
                score += 5
        print(
            "player "
            + str(self.current_player)
            + "scored "
            + str(score)
            + " points with fields!"
        )
        return score

    def _score_buildings(self, board):
        score = 0
        for coord, stack in board.items():
            if not stack:  # If stack is empty, skip
                continue

            # Only assign top and h if stack is NOT empty
            top = stack[-1]
            h = len(stack)

            if top == BUILDING and h == 2:
                n_types = set(
                    self._get_top_tile(board, nc) for nc in get_neighbors(coord)
                )
                n_types.discard(None)
                if len(n_types) >= 3:
                    score += 5
        print(
            "player "
            + str(self.current_player)
            + "scored "
            + str(score)
            + " points with buildings!"
        )
        return score

    def _score_water(self, board):
        score = 0
        visited = set()
        waters = [c for c, s in board.items() if self._get_top_tile(board, c) == WATER]
        for start in waters:
            if start in visited:
                continue
            comp_coords = set()
            q_comp = deque([start])
            visited.add(start)
            comp_coords.add(start)
            while q_comp:
                curr = q_comp.popleft()
                for n in get_neighbors(curr):
                    if n not in visited and self._get_top_tile(board, n) == WATER:
                        visited.add(n)
                        comp_coords.add(n)
                        q_comp.append(n)
            if len(comp_coords) >= 2:
                comp_list = list(comp_coords)

                def get_comp_neighbors(node):
                    return [nei for nei in get_neighbors(node) if nei in comp_coords]

                diameter = 0
                for i in range(len(comp_list)):
                    node1 = comp_list[i]
                    q_bfs = deque([(node1, 0)])
                    visited_bfs = {node1}
                    max_dist = 0
                    while q_bfs:
                        curr_b, dist_b = q_bfs.popleft()
                        max_dist = max(max_dist, dist_b)
                        for neighbor_b in get_comp_neighbors(curr_b):
                            if neighbor_b not in visited_bfs:
                                visited_bfs.add(neighbor_b)
                                q_bfs.append((neighbor_b, dist_b + 1))
                    diameter = max(diameter, max_dist)
                score += get_water_score(diameter + 1)
        print(
            "player "
            + str(self.current_player)
            + "scored "
            + str(score)
            + " points with water!"
        )
        return score

    # --- clone (unchanged) ---
    def clone(self):
        return copy.deepcopy(self)

    # --- __str__ (unchanged logic, uses correct count) ---
    def __str__(self):
        s = f"--- Harmonies State (Grid: 5-4-5-4-5 rows) ---\n"
        s += f"Player Turn: {self.current_player}, Phase: {self.turn_phase}\n"
        s += f"Game Over: {self.is_game_over()}, Winner: {self.winner}, Scores: {self.final_scores}\n"
        s += f"Bag: {dict(sorted(self.tile_bag.items()))}\n"
        s += f"Available Piles: {self.available_piles}\n"
        s += f"Player {self.current_player} Hand: {self.tiles_in_hand}\n"
        for p in [0, 1]:
            s += f"Player {p} Board ({len(self.player_boards[p])}/{PLAYER_BOARD_HEX_COUNT} hexes):\n"
            board_repr = {
                str(coord): stack
                for coord, stack in sorted(self.player_boards[p].items())
            }
            s += f"  {board_repr}\n"
        s += "---------------------------------------------\n"
        return s


# # --- Example Usage ---
# if __name__ == "__main__":
#     print("--- Initializing Game with 5-4-5-4-5 Row Grid ---")
#     # Optional: Print the generated hex coordinates to verify
#     print(f"Generated {len(VALID_HEXES)} hex coordinates:")
#     # print(sorted(list(VALID_HEXES))) # Uncomment to see all coords

#     # Test neighbors of a few key hexes
#     print("\nNeighbor Checks:")
#     print(f"Neighbors of center (0,0): {sorted(get_neighbors((0,0)))}") # Should have 6 neighbors
#     print(f"Neighbors of top-left (-1,-2): {sorted(get_neighbors((-1,-2)))}") # Corner, should have fewer
#     print(f"Neighbors of middle-left (-2,0): {sorted(get_neighbors((-2,0)))}") # Edge, should have fewer
#     print(f"Neighbors of indented edge (-1,-1): {sorted(get_neighbors((-1,-1)))}") # Should have 6 neighbors
#     print(f"Neighbors of indented edge (1,1): {sorted(get_neighbors((1,1)))}") # Should have 6 neighbors
#     print("-" * 20)


#     game = HarmoniesGameState()
#     print("\nInitial State:")
#     print(game)

#     # --- Random Game Simulation (identical structure to before) ---
#     turn_count = 0; max_turns = 150
#     print("\n--- Starting Random Game Simulation ---")
#     while not game.is_game_over() and turn_count < max_turns:
#         # (Rest of random loop...)
#         # print(f"\n--- Turn {turn_count // 6 + 1} (Action {turn_count % 6 + 1}) ---") # Verbose
#         # print(f"Player: {game.get_current_player()}, Phase: {game.turn_phase}")
#         legal_moves = game.get_legal_moves()
#         if not legal_moves:
#             print("No legal moves available!")
#             if game.turn_phase == "choose_pile" and not game.available_piles and sum(game.tile_bag.values()) == 0:
#                  if not game.game_over:
#                      print("Forcing score calculation..."); game.game_over = True ; game.turn_phase = "game_over"
#                      game._calculate_final_scores() ; game._determine_winner()
#             break
#         chosen_move = random.choice(legal_moves)
#         # print(f"Player {game.get_current_player()} chooses move: {chosen_move}") # Verbose
#         try:
#             game = game.apply_move(chosen_move)
#             turn_count += 1
#         except ValueError as e:
#             print(f"Error: {e}\nCurrent State:\n{game}\nMove Attempted: {chosen_move}\nLegal Moves: {legal_moves}")
#             break

#     print("\n" + "="*20 + "\nGame Over!\n" + "="*20)
#     print(f"Final State (after {turn_count} actions):"); print(game)

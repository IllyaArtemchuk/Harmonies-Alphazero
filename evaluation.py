import random
import time
from tqdm import tqdm
from harmonies_engine import HarmoniesGameState


def run_tournament(
    num_games, az_agent_func, greedy_agent_func, az_args, greedy_args=None
):
    """
    Runs multiple games between AlphaZero and Greedy agents.

    Args:
        num_games (int): Total number of games to play (should be even).
        az_agent_func: Function for AlphaZero agent's move.
        greedy_agent_func: Function for Greedy agent's move.
        az_args (tuple): Arguments needed by az_agent_func.
        greedy_args (tuple, optional): Arguments needed by greedy_agent_func (likely None).
    """
    if num_games % 2 != 0:
        print("Warning: Number of games should be even for fair player assignment.")
        num_games += 1  # Play one extra if odd

    az_wins = 0
    greedy_wins = 0
    draws = 0

    print(f"\n--- Starting Tournament: AlphaZero vs Greedy ({num_games} games) ---")

    for i in tqdm(range(num_games), desc="Tournament Games"):
        if i % 2 == 0:
            # AlphaZero plays as Player 0
            print(f"\nGame {i+1}: AlphaZero (P0) vs Greedy (P1)")
            outcome = play_game(az_agent_func, greedy_agent_func, az_args, greedy_args)
            if outcome == 1:
                az_wins += 1
            elif outcome == -1:
                greedy_wins += 1
            else:
                draws += 1
        else:
            # Greedy plays as Player 0
            print(f"\nGame {i+1}: Greedy (P0) vs AlphaZero (P1)")
            outcome = play_game(greedy_agent_func, az_agent_func, greedy_args, az_args)
            if outcome == 1:  # Greedy (P0) won
                greedy_wins += 1
            elif outcome == -1:  # AlphaZero (P1) won
                az_wins += 1
            else:
                draws += 1

        print(f"Game {i+1} Result: {outcome} (1=P0 Win, -1=P1 Win, 0=Draw/Error)")

    print("\n--- Tournament Finished ---")
    print(f"Results over {num_games} games:")
    print(f"  AlphaZero Wins: {az_wins}")
    print(f"  Greedy Wins:    {greedy_wins}")
    print(f"  Draws/Errors:   {draws}")

    total_non_draws = az_wins + greedy_wins
    if total_non_draws > 0:
        az_win_rate = az_wins / total_non_draws
        print(f"  AlphaZero Win Rate (vs Greedy, excluding draws): {az_win_rate:.3f}")
    else:
        print("  No decisive games played.")


def play_game(player0_func, player1_func, args0=None, args1=None):
    """
    Plays a single game between two agents.

    Args:
        player0_func: Function to call for Player 0's move (e.g., get_best_action_and_pi).
        player1_func: Function to call for Player 1's move (e.g., choose_move_greedy).
        args0: Tuple of additional arguments needed by player0_func (e.g., model_manager, config).
        args1: Tuple of additional arguments needed by player1_func (e.g., model_manager, config).

    Returns:
        int: 1 if Player 0 wins, -1 if Player 1 wins, 0 for draw/error.
    """
    game = HarmoniesGameState()
    players = {0: player0_func, 1: player1_func}
    player_args = {
        0: args0 if args0 is not None else (),
        1: args1 if args1 is not None else (),
    }

    while not game.is_game_over():
        current_player = game.get_current_player()
        move_function = players[current_player]
        current_args = player_args[current_player]

        # Prepare arguments for the move function
        # Standard args are game_state, then others packed in a tuple
        move_args = (game.clone(),) + current_args

        try:
            # Call the appropriate function to get the move
            # get_best_action_and_pi returns (action, pi), greedy returns action
            result = move_function(*move_args)

            if isinstance(result, tuple):  # AZ agent likely returned (action, pi)
                best_action = result[0]
            else:
                best_action = result

            if best_action is None:
                print(f"ERROR: Player {current_player}'s agent returned None action.")
                return 0  # Treat as error/draw

        except Exception as e:
            print(
                f"ERROR: Exception during Player {current_player}'s move function: {e}"
            )
            print(f"State:\n{game}")
            # import traceback; traceback.print_exc() # For detailed debug
            return 0  # Treat as error/draw

        # Apply the chosen move
        try:
            game = game.apply_move(best_action)
        except Exception as e:
            print(
                f"ERROR: Exception during apply_move for Player {current_player}: {e}"
            )
            print(f"State before move:\n{game}")  # Show previous state
            print(f"Action attempted: {best_action}")
            return 0  # Treat as error/draw

    # Game finished
    outcome = game.get_game_outcome()
    if outcome is None:
        print("ERROR: Game finished but outcome is None.")
        return 0
    return outcome


def choose_move_greedy(game_state: HarmoniesGameState):
    """
    Selects the legal move that leads to the immediate next state
    with the highest score for the current player.
    Handles both 'choose_pile' and 'place_tile' phases somewhat naively.
    """
    current_player = game_state.get_current_player()
    legal_moves = game_state.get_legal_moves()

    if not legal_moves:
        return None  # No moves possible

    best_move = None
    # Initialize with a very low score to ensure any valid score is better
    best_score = -float("inf")

    # Evaluate each legal move
    for move in legal_moves:
        try:
            # Simulate applying the move to get the next state
            # IMPORTANT: Assumes apply_move returns a NEW state object
            next_state = game_state.apply_move(move)

            # We use the player from the *original* state because apply_move might change it.
            current_board = next_state.player_boards[current_player]
            score = next_state._calculate_score_for_player(
                current_player
            )  # Check if this recalculates full score

            # If this move yields a better score, update best move
            if score > best_score:
                best_score = score
                best_move = move

        except Exception as e:
            # Log error if simulating a move fails
            print(f"GREEDY AGENT ERROR: Failed evaluating move {move}: {e}")
            continue  # Skip this move

    # If no move improved the score (or all moves failed), pick randomly
    if best_move is None:
        print("GREEDY AGENT WARNING: No best move found, choosing randomly.")
        best_move = random.choice(legal_moves)

    return best_move

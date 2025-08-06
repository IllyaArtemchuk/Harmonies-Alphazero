import random
import time
from tqdm import tqdm
from connect4_engine import Connect4GameState


def run_connect4_tournament(
    num_games, az_agent_func, greedy_agent_func, az_args, greedy_args=None
):
    """
    Runs multiple Connect 4 games between AlphaZero and Greedy agents.

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

    print(f"\n--- Starting Connect 4 Tournament: AlphaZero vs Greedy ({num_games} games) ---")

    for i in tqdm(range(num_games), desc="Connect 4 Tournament Games"):
        if i % 2 == 0:
            # AlphaZero plays as Player 0
            print(f"\nGame {i+1}: AlphaZero (P0) vs Greedy (P1)")
            outcome = play_connect4_game(az_agent_func, greedy_agent_func, az_args, greedy_args)
            if outcome == 1:
                az_wins += 1
            elif outcome == -1:
                greedy_wins += 1
            else:
                draws += 1
        else:
            # Greedy plays as Player 0
            print(f"\nGame {i+1}: Greedy (P0) vs AlphaZero (P1)")
            outcome = play_connect4_game(greedy_agent_func, az_agent_func, greedy_args, az_args)
            if outcome == 1:  # Greedy (P0) won
                greedy_wins += 1
            elif outcome == -1:  # AlphaZero (P1) won
                az_wins += 1
            else:
                draws += 1

        print(f"Game {i+1} Result: {outcome} (1=P0 Win, -1=P1 Win, 0=Draw/Error)")

    print("\n--- Connect 4 Tournament Finished ---")
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


def play_connect4_game(player0_func, player1_func, args0=None, args1=None):
    """
    Plays a single Connect 4 game between two agents.

    Args:
        player0_func: Function to call for Player 0's move (e.g., get_best_action_and_pi).
        player1_func: Function to call for Player 1's move (e.g., choose_move_greedy_connect4).
        args0: Tuple of additional arguments needed by player0_func (e.g., model_manager, config).
        args1: Tuple of additional arguments needed by player1_func (e.g., model_manager, config).

    Returns:
        int: 1 if Player 0 wins, -1 if Player 1 wins, 0 for draw/error.
    """
    game = Connect4GameState()
    players = {0: player0_func, 1: player1_func}
    player_args = {
        0: args0 if args0 is not None else (),
        1: args1 if args1 is not None else (),
    }

    move_count = 0
    while not game.is_game_over() and move_count < 42:  # Max possible moves in Connect 4
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
            print(f"State:\n")
            game.render()
            return 0  # Treat as error/draw

        # Apply the chosen move
        try:
            game = game.apply_move(best_action)
            move_count += 1
        except Exception as e:
            print(
                f"ERROR: Exception during apply_move for Player {current_player}: {e}"
            )
            print(f"State before move:")
            game.render()
            print(f"Action attempted: {best_action}")
            return 0  # Treat as error/draw

    # Game finished
    outcome = game.get_game_outcome()
    if outcome is None:
        print("ERROR: Game finished but outcome is None.")
        return 0
    return outcome


def choose_move_greedy_connect4(game_state: Connect4GameState):
    """
    Connect 4 greedy agent that prioritizes:
    1. Winning moves
    2. Blocking opponent wins
    3. Center columns
    4. Random legal move
    """
    current_player = game_state.get_current_player()
    legal_moves = game_state.get_legal_moves()

    if not legal_moves:
        print("GREEDY AGENT WARNING: No legal moves available.")
        return None

    # 1. Check for winning moves
    for move in legal_moves:
        try:
            next_state = game_state.apply_move(move)
            if next_state.is_game_over() and next_state.winner == current_player:
                return move
        except Exception:
            continue

    # 2. Check for blocking moves (opponent would win)
    opponent = 1 - current_player
    for move in legal_moves:
        try:
            # Simulate opponent playing this move
            temp_state = game_state.clone()
            temp_state.current_player = opponent
            next_state = temp_state.apply_move(move)
            if next_state.is_game_over() and next_state.winner == opponent:
                return move  # Block this winning move
        except Exception:
            continue

    # 3. Prefer center columns (they offer more winning opportunities)
    center_preference = [3, 2, 4, 1, 5, 0, 6]  # Center to edges
    for preferred_col in center_preference:
        if preferred_col in legal_moves:
            return preferred_col

    # 4. Fallback to random legal move
    return random.choice(legal_moves)


def evaluate_connect4_position(game_state: Connect4GameState, player: int):
    """
    Simple position evaluation for Connect 4.
    Returns a score indicating how good the position is for the given player.
    """
    if game_state.is_game_over():
        if game_state.winner == player:
            return 1000  # Win
        elif game_state.winner is not None:
            return -1000  # Loss
        else:
            return 0  # Draw

    score = 0
    board = game_state.board
    player_piece = player + 1
    opponent_piece = (1 - player) + 1

    # Evaluate all possible 4-in-a-row positions
    for row in range(6):
        for col in range(7):
            # Check horizontal
            if col <= 3:
                window = [board[row, col + i] for i in range(4)]
                score += evaluate_window(window, player_piece, opponent_piece)
            
            # Check vertical
            if row <= 2:
                window = [board[row + i, col] for i in range(4)]
                score += evaluate_window(window, player_piece, opponent_piece)
            
            # Check diagonal (positive slope)
            if row <= 2 and col <= 3:
                window = [board[row + i, col + i] for i in range(4)]
                score += evaluate_window(window, player_piece, opponent_piece)
            
            # Check diagonal (negative slope)
            if row >= 3 and col <= 3:
                window = [board[row - i, col + i] for i in range(4)]
                score += evaluate_window(window, player_piece, opponent_piece)

    return score


def evaluate_window(window, piece, opponent_piece):
    """Evaluate a 4-piece window for Connect 4."""
    score = 0
    piece_count = window.count(piece)
    opponent_count = window.count(opponent_piece)
    empty_count = window.count(0)

    if piece_count == 4:
        score += 100
    elif piece_count == 3 and empty_count == 1:
        score += 10
    elif piece_count == 2 and empty_count == 2:
        score += 2

    if opponent_count == 3 and empty_count == 1:
        score -= 80  # Block opponent

    return score 
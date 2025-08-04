from connect4_engine import Connect4GameState
from process_connect4_state import create_state_tensors, get_action_size
import random

def test_basic_game():
    print("Testing basic Connect 4 game mechanics...")
    game = Connect4GameState()
    
    print("\nInitial board:")
    game.render()
    
    print("\nLegal moves:", game.get_legal_moves())
    print("Current player:", game.get_current_player())
    print("Game over?", game.is_game_over())
    
    print("\n\nMaking some moves...")
    moves = [3, 3, 4, 4, 5, 5]
    for i, move in enumerate(moves):
        print(f"\nMove {i+1}: Player {game.get_current_player()} plays column {move}")
        game = game.apply_move(move)
        game.render()
    
    print("\nMaking winning move...")
    game = game.apply_move(6)
    game.render()
    
    print(f"\nGame over: {game.is_game_over()}")
    print(f"Winner: {game.winner}")
    print(f"Outcome: {game.get_game_outcome()}")

def test_draw():
    print("\n\n=== Testing draw scenario ===")
    game = Connect4GameState()
    
    moves = []
    for col in range(7):
        for _ in range(6):
            moves.append(col)
    
    random.shuffle(moves[:21])
    random.shuffle(moves[21:])
    
    move_count = 0
    for move in moves:
        if move in game.get_legal_moves():
            game = game.apply_move(move)
            move_count += 1
            
            if game.is_game_over():
                break
    
    print(f"Played {move_count} moves")
    game.render()
    print(f"Game over: {game.is_game_over()}")
    print(f"Winner: {game.winner}")
    print(f"Outcome: {game.get_game_outcome()}")

def test_state_tensors():
    print("\n\n=== Testing state tensor creation ===")
    game = Connect4GameState()
    
    game = game.apply_move(3)
    game = game.apply_move(3)
    game = game.apply_move(4)
    
    board_tensor, global_features = create_state_tensors(game)
    
    print(f"Board tensor shape: {board_tensor.shape}")
    print(f"Global features shape: {global_features.shape}")
    print(f"Action size: {get_action_size()}")
    
    print("\nBoard tensor channel 0 (current player pieces):")
    print(board_tensor[0])
    print("\nBoard tensor channel 1 (opponent pieces):")
    print(board_tensor[1])
    print("\nGlobal features:", global_features)

def test_clone_and_hash():
    print("\n\n=== Testing clone and hash ===")
    game1 = Connect4GameState()
    game1 = game1.apply_move(3)
    
    game2 = game1.clone()
    game2 = game2.apply_move(4)
    
    print(f"Game 1 hash: {hash(game1)}")
    print(f"Game 2 hash: {hash(game2)}")
    print(f"Hashes equal? {hash(game1) == hash(game2)}")
    
    game3 = game1.clone()
    print(f"Game 3 (clone of game1) hash: {hash(game3)}")
    print(f"Game 1 and Game 3 hashes equal? {hash(game1) == hash(game3)}")

if __name__ == "__main__":
    test_basic_game()
    test_draw()
    test_state_tensors()
    test_clone_and_hash()
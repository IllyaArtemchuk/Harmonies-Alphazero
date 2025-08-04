import torch
from game_config import GameState, create_state_tensors, ACTION_SIZE, BOARD_TENSOR_SHAPE, GLOBAL_FEATURES_SIZE

def test_game_config():
    print("Testing game configuration...")
    print(f"Game type: Connect4")
    print(f"Action size: {ACTION_SIZE}")
    print(f"Board tensor shape: {BOARD_TENSOR_SHAPE}")
    print(f"Global features size: {GLOBAL_FEATURES_SIZE}")
    
    # Test game creation
    game = GameState()
    print(f"\nGame created successfully")
    print(f"Current player: {game.get_current_player()}")
    print(f"Legal moves: {game.get_legal_moves()}")
    
    # Test state tensor creation
    board_tensor, global_features = create_state_tensors(game)
    print(f"\nState tensors created:")
    print(f"Board tensor shape: {board_tensor.shape}")
    print(f"Global features shape: {global_features.shape}")
    
    # Test a few moves
    game = game.apply_move(3)
    game = game.apply_move(4)
    print(f"\nAfter 2 moves:")
    print(f"Current player: {game.get_current_player()}")
    print(f"Legal moves: {game.get_legal_moves()}")
    
    return True

def test_model_creation():
    print("\n\nTesting model creation...")
    from config import model_config_default, training_config_default
    from model import ModelManager
    
    print(f"Model config: {model_config_default}")
    
    try:
        model_mgr = ModelManager(model_config_default, training_config_default)
        print("Model created successfully!")
        
        # Test a forward pass
        game = GameState()
        board_tensor, global_features = create_state_tensors(game)
        
        # Add batch dimension and move to device
        board_tensor = board_tensor.unsqueeze(0).to(model_mgr.device)
        global_features = global_features.unsqueeze(0).to(model_mgr.device)
        
        with torch.no_grad():
            policy, value = model_mgr.model(board_tensor, global_features)
        
        print(f"Forward pass successful!")
        print(f"Policy shape: {policy.shape}")
        print(f"Value shape: {value.shape}")
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Connect 4 Integration ===\n")
    
    if test_game_config():
        print("\n✓ Game configuration test passed")
    else:
        print("\n✗ Game configuration test failed")
    
    if test_model_creation():
        print("\n✓ Model creation test passed")
    else:
        print("\n✗ Model creation test failed")
    
    print("\n=== All tests completed ===")
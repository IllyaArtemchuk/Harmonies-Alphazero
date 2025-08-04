"""
Example of how to use the enhanced metrics and monitoring during training.
This shows how to integrate the metrics into your training loop for deep insights.
"""

from training_metrics import TrainingMetrics
from training_dashboard import create_training_monitor
import numpy as np
import time

def example_training_with_metrics():
    """
    Example showing how metrics provide insights during training.
    """
    
    # Initialize metrics tracking
    metrics = TrainingMetrics(experiment_name="connect4_example")
    monitor = create_training_monitor("connect4_example")
    
    print("Starting example training with enhanced metrics...")
    print(f"\nTensorBoard command: tensorboard --logdir {metrics.log_dir.parent}")
    print("\nMetrics will show:")
    print("- Real-time loss curves")
    print("- Win rate progression") 
    print("- Policy entropy (exploration vs exploitation)")
    print("- Model weight distributions")
    print("- Game-specific insights\n")
    
    # Simulate training iterations
    for iteration in range(10):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Simulate self-play phase
        print("Running self-play...")
        game_data = []
        for _ in range(20):  # 20 games
            # Simulate a game
            game_length = np.random.randint(10, 42)
            outcome = np.random.choice([-1, 0, 1])
            moves = np.random.randint(0, 7, size=game_length).tolist()
            
            game_data.append({
                'length': game_length,
                'outcome': outcome,
                'moves': moves
            })
        
        # Log self-play metrics
        metrics.log_self_play_metrics(game_data)
        
        # Simulate training phase
        print("Training model...")
        for batch in range(10):
            # Simulate losses
            total_loss = 0.5 * np.exp(-iteration * 0.1) + np.random.normal(0, 0.1)
            policy_loss = 0.3 * np.exp(-iteration * 0.1) + np.random.normal(0, 0.05)
            value_loss = 0.2 * np.exp(-iteration * 0.1) + np.random.normal(0, 0.05)
            
            losses = {
                'total': max(0, total_loss),
                'policy': max(0, policy_loss),
                'value': max(0, value_loss)
            }
            
            metrics.log_training_step(losses, batch)
        
        # Log epoch summary
        epoch_metrics = {
            'avg_total_loss': total_loss,
            'avg_policy_loss': policy_loss,
            'avg_value_loss': value_loss,
            'batches_processed': 10,
            'training_time': 5.0
        }
        metrics.log_epoch_summary(epoch_metrics)
        
        # Simulate evaluation
        if iteration % 2 == 0:
            print("Evaluating model...")
            win_rate = 0.5 + iteration * 0.05 + np.random.normal(0, 0.1)
            win_rate = np.clip(win_rate, 0, 1)
            metrics.log_evaluation_results(win_rate, games_played=20)
            
            print(f"Win rate vs previous best: {win_rate:.1%}")
        
        # Update dashboard monitor
        monitor.update({
            'loss': total_loss,
            'win_rate': win_rate if iteration % 2 == 0 else None,
            'game_length': np.mean([g['length'] for g in game_data]),
            'entropy': 0.7 * np.exp(-iteration * 0.05)  # Simulated entropy decay
        }, game_data)
        
        # Print insights periodically
        if iteration % 3 == 0:
            monitor.print_full_status()
        
        # Log value prediction accuracy
        predictions = np.random.uniform(-1, 1, 50)
        actuals = predictions + np.random.normal(0, 0.2, 50)
        actuals = np.clip(actuals, -1, 1)
        metrics.log_value_prediction_accuracy(predictions.tolist(), actuals.tolist())
        
        # Log policy entropy
        entropies = [0.7 * np.exp(-iteration * 0.05) + np.random.normal(0, 0.05) for _ in range(20)]
        metrics.log_policy_entropy(entropies)
        
        metrics.increment_iteration()
        time.sleep(0.5)  # Simulate computation time
    
    # Final report
    print("\n\n=== TRAINING COMPLETE ===")
    monitor.print_full_status()
    monitor.save_report()
    
    # Close metrics
    metrics.close()
    
    print("\n\nKey insights from metrics:")
    print("1. Loss curves show if model is learning (decreasing) or diverging")
    print("2. Win rate progression shows actual game performance improvement")
    print("3. Policy entropy decay shows model becoming more confident")
    print("4. Value prediction accuracy shows if model understands position evaluation")
    print("5. Connect 4 insights show if model learns optimal strategies (center column preference)")
    print("\nCheck TensorBoard for detailed visualizations!")

if __name__ == "__main__":
    example_training_with_metrics()
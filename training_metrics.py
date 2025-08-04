"""
Enhanced training metrics and monitoring for AlphaZero.
Provides TensorBoard integration and advanced metrics collection.
"""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

class TrainingMetrics:
    def __init__(self, log_dir: str = "./runs", experiment_name: Optional[str] = None):
        """
        Initialize metrics tracking with TensorBoard support.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this training run
        """
        if experiment_name is None:
            experiment_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.iteration = 0
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.game_metrics = GameMetrics()
        self.model_metrics = ModelMetrics()
        
        # Moving averages for smoothing
        self.loss_ema = ExponentialMovingAverage(alpha=0.99)
        self.win_rate_ema = ExponentialMovingAverage(alpha=0.95)
        
    def log_training_step(self, losses: Dict[str, float], batch_idx: int):
        """Log metrics from a single training batch."""
        # Update EMAs
        self.loss_ema.update(losses['total'])
        
        # Log to TensorBoard
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(f'Loss/{loss_name}', loss_value, self.global_step)
        
        # Log smoothed loss
        self.writer.add_scalar('Loss/total_smoothed', self.loss_ema.value, self.global_step)
        
        self.global_step += 1
        
    def log_epoch_summary(self, epoch_metrics: Dict[str, float]):
        """Log summary metrics at the end of a training epoch."""
        for metric_name, value in epoch_metrics.items():
            self.writer.add_scalar(f'Epoch/{metric_name}', value, self.iteration)
            self.metrics_history[metric_name].append(value)
    
    def log_learning_rate(self, lr: float):
        """Log current learning rate."""
        self.writer.add_scalar('Training/learning_rate', lr, self.global_step)
    
    def log_evaluation_results(self, win_rate: float, games_played: int):
        """Log evaluation results against previous best model."""
        self.win_rate_ema.update(win_rate)
        
        self.writer.add_scalar('Evaluation/win_rate', win_rate, self.iteration)
        self.writer.add_scalar('Evaluation/win_rate_smoothed', self.win_rate_ema.value, self.iteration)
        self.writer.add_scalar('Evaluation/games_played', games_played, self.iteration)
        
        self.metrics_history['eval_win_rate'].append(win_rate)
    
    def log_self_play_metrics(self, games_data: List[Dict[str, Any]]):
        """Log metrics from self-play games."""
        if not games_data:
            return
            
        # Aggregate metrics
        game_lengths = [g['length'] for g in games_data]
        outcomes = [g['outcome'] for g in games_data]
        
        avg_length = np.mean(game_lengths)
        win_rate_player0 = sum(1 for o in outcomes if o == 1) / len(outcomes)
        draw_rate = sum(1 for o in outcomes if o == 0) / len(outcomes)
        
        self.writer.add_scalar('SelfPlay/avg_game_length', avg_length, self.iteration)
        self.writer.add_scalar('SelfPlay/player0_win_rate', win_rate_player0, self.iteration)
        self.writer.add_scalar('SelfPlay/draw_rate', draw_rate, self.iteration)
        
        # Histogram of game lengths
        self.writer.add_histogram('SelfPlay/game_lengths', np.array(game_lengths), self.iteration)
    
    def log_mcts_metrics(self, mcts_stats: Dict[str, float]):
        """Log MCTS performance metrics."""
        for stat_name, value in mcts_stats.items():
            self.writer.add_scalar(f'MCTS/{stat_name}', value, self.global_step)
    
    def log_model_weights(self, model: torch.nn.Module, log_gradients: bool = True):
        """Log model weight statistics and gradients."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Weights
                self.writer.add_histogram(f'Weights/{name}', param.data, self.iteration)
                self.writer.add_scalar(f'Weights/{name}_mean', param.data.mean(), self.iteration)
                self.writer.add_scalar(f'Weights/{name}_std', param.data.std(), self.iteration)
                
                # Gradients
                if log_gradients and param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, self.iteration)
                    self.writer.add_scalar(f'Gradients/{name}_norm', param.grad.norm(), self.iteration)
    
    def log_replay_buffer_stats(self, buffer_stats: Dict[str, Any]):
        """Log replay buffer statistics."""
        self.writer.add_scalar('ReplayBuffer/size', buffer_stats['size'], self.iteration)
        self.writer.add_scalar('ReplayBuffer/capacity', buffer_stats['capacity'], self.iteration)
        
        if 'age_distribution' in buffer_stats:
            self.writer.add_histogram('ReplayBuffer/sample_ages', 
                                    buffer_stats['age_distribution'], self.iteration)
    
    def log_value_prediction_accuracy(self, predictions: List[float], actuals: List[float]):
        """Log value prediction accuracy metrics."""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        self.writer.add_scalar('Value/mse', mse, self.iteration)
        self.writer.add_scalar('Value/mae', mae, self.iteration)
        
        # Calibration plot data
        self.writer.add_histogram('Value/predictions', predictions, self.iteration)
        self.writer.add_histogram('Value/actuals', actuals, self.iteration)
    
    def log_policy_entropy(self, entropies: List[float]):
        """Log policy entropy to track exploration."""
        avg_entropy = np.mean(entropies)
        self.writer.add_scalar('Policy/avg_entropy', avg_entropy, self.iteration)
        self.writer.add_histogram('Policy/entropy_distribution', np.array(entropies), self.iteration)
    
    def save_iteration_summary(self):
        """Save a summary of the current iteration to JSON."""
        summary = {
            'iteration': self.iteration,
            'global_step': self.global_step,
            'timestamp': datetime.now().isoformat(),
            'metrics': {k: v[-1] if v else None for k, v in self.metrics_history.items()}
        }
        
        summary_path = self.log_dir / f'iteration_{self.iteration:04d}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def increment_iteration(self):
        """Move to next iteration."""
        self.iteration += 1
        self.save_iteration_summary()
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class GameMetrics:
    """Track game-specific metrics during self-play and evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.game_lengths = []
        self.outcomes = []
        self.move_counts = defaultdict(int)
        self.position_values = []
        self.policy_entropies = []
    
    def add_game(self, game_data: Dict[str, Any]):
        """Add metrics from a completed game."""
        self.game_lengths.append(game_data.get('length', 0))
        self.outcomes.append(game_data.get('outcome', 0))
        
        # Track move frequencies
        if 'moves' in game_data:
            for move in game_data['moves']:
                self.move_counts[move] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.game_lengths:
            return {}
        
        return {
            'avg_game_length': np.mean(self.game_lengths),
            'std_game_length': np.std(self.game_lengths),
            'win_rate_player0': sum(1 for o in self.outcomes if o == 1) / len(self.outcomes),
            'draw_rate': sum(1 for o in self.outcomes if o == 0) / len(self.outcomes),
            'total_games': len(self.game_lengths),
            'most_common_moves': sorted(self.move_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]
        }


class ModelMetrics:
    """Track model-specific metrics."""
    
    def __init__(self):
        self.value_predictions = []
        self.value_targets = []
        self.policy_accuracies = []
        
    def add_predictions(self, pred_values: List[float], true_values: List[float]):
        """Add value predictions and their targets."""
        self.value_predictions.extend(pred_values)
        self.value_targets.extend(true_values)
    
    def calculate_calibration(self) -> Dict[str, float]:
        """Calculate value calibration metrics."""
        if not self.value_predictions:
            return {}
        
        preds = np.array(self.value_predictions)
        targets = np.array(self.value_targets)
        
        # Bin predictions and calculate calibration
        bins = np.linspace(-1, 1, 11)
        calibration = []
        
        for i in range(len(bins) - 1):
            mask = (preds >= bins[i]) & (preds < bins[i + 1])
            if mask.sum() > 0:
                avg_pred = preds[mask].mean()
                avg_true = targets[mask].mean()
                calibration.append({
                    'bin_center': (bins[i] + bins[i + 1]) / 2,
                    'avg_prediction': avg_pred,
                    'avg_outcome': avg_true,
                    'count': mask.sum()
                })
        
        return calibration


class ExponentialMovingAverage:
    """Simple EMA for smoothing metrics."""
    
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value = None
        
    def update(self, new_value: float):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
    
    def reset(self):
        self.value = None


class Connect4Metrics:
    """Connect 4 specific metrics tracking."""
    
    @staticmethod
    def analyze_game(game_history: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze a Connect 4 game for specific patterns."""
        metrics = {
            'opening_move': game_history[0][1] if game_history else None,
            'center_column_plays': sum(1 for _, move in game_history if move == 3),
            'game_length': len(game_history),
        }
        
        # Check if perfect play (center column opening)
        if game_history and game_history[0][1] == 3:
            metrics['perfect_opening'] = True
        else:
            metrics['perfect_opening'] = False
            
        return metrics
    
    @staticmethod
    def get_winning_pattern(board: np.ndarray, last_move: Tuple[int, int]) -> Optional[str]:
        """Identify the winning pattern type."""
        # Implementation would check for horizontal, vertical, diagonal wins
        # Returns: 'horizontal', 'vertical', 'diagonal_up', 'diagonal_down', or None
        pass
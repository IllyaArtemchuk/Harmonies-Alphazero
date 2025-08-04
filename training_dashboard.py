"""
Real-time training dashboard for monitoring AlphaZero training progress.
Provides insights into model learning and performance.
"""

import time
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional

class TrainingDashboard:
    """
    Provides real-time monitoring and analysis of training progress.
    """
    
    def __init__(self, log_dir: str = "./dashboard_logs", window_size: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Moving windows for real-time metrics
        self.loss_window = deque(maxlen=window_size)
        self.win_rate_window = deque(maxlen=window_size)
        self.game_length_window = deque(maxlen=window_size)
        self.entropy_window = deque(maxlen=window_size)
        
        # Tracking thresholds for alerts
        self.loss_explosion_threshold = 10.0
        self.stagnation_threshold = 50  # iterations without improvement
        self.min_entropy_threshold = 0.1
        
        # State tracking
        self.iterations_without_improvement = 0
        self.best_win_rate = 0.0
        self.alerts = []
        
    def update_metrics(self, metrics: Dict):
        """Update dashboard with new metrics."""
        timestamp = datetime.now()
        
        # Update moving windows
        if 'loss' in metrics:
            self.loss_window.append(metrics['loss'])
            self._check_loss_health(metrics['loss'])
            
        if 'win_rate' in metrics:
            self.win_rate_window.append(metrics['win_rate'])
            self._check_progress(metrics['win_rate'])
            
        if 'game_length' in metrics:
            self.game_length_window.append(metrics['game_length'])
            
        if 'entropy' in metrics:
            self.entropy_window.append(metrics['entropy'])
            self._check_exploration(metrics['entropy'])
        
        # Save snapshot
        self._save_snapshot(timestamp, metrics)
        
    def _check_loss_health(self, loss: float):
        """Check for loss explosion or NaN values."""
        if np.isnan(loss) or np.isinf(loss):
            self.alerts.append({
                'type': 'critical',
                'message': 'NaN or Inf loss detected!',
                'timestamp': datetime.now()
            })
        elif loss > self.loss_explosion_threshold:
            self.alerts.append({
                'type': 'warning',
                'message': f'Loss explosion detected: {loss:.4f}',
                'timestamp': datetime.now()
            })
    
    def _check_progress(self, win_rate: float):
        """Check for training stagnation."""
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
            
        if self.iterations_without_improvement > self.stagnation_threshold:
            self.alerts.append({
                'type': 'warning',
                'message': f'Training stagnation: {self.iterations_without_improvement} iterations without improvement',
                'timestamp': datetime.now()
            })
    
    def _check_exploration(self, entropy: float):
        """Check if model is still exploring."""
        if entropy < self.min_entropy_threshold:
            self.alerts.append({
                'type': 'info',
                'message': f'Low policy entropy: {entropy:.4f} - model may be overfitting',
                'timestamp': datetime.now()
            })
    
    def get_summary(self) -> Dict:
        """Get current training summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'avg_loss': np.mean(self.loss_window) if self.loss_window else None,
                'loss_std': np.std(self.loss_window) if self.loss_window else None,
                'avg_win_rate': np.mean(self.win_rate_window) if self.win_rate_window else None,
                'avg_game_length': np.mean(self.game_length_window) if self.game_length_window else None,
                'avg_entropy': np.mean(self.entropy_window) if self.entropy_window else None,
            },
            'health': {
                'iterations_without_improvement': self.iterations_without_improvement,
                'best_win_rate': self.best_win_rate,
                'recent_alerts': self.alerts[-10:]  # Last 10 alerts
            }
        }
        return summary
    
    def _save_snapshot(self, timestamp: datetime, metrics: Dict):
        """Save metrics snapshot to file."""
        snapshot_file = self.log_dir / f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump({
                'timestamp': timestamp.isoformat(),
                'metrics': metrics,
                'summary': self.get_summary()
            }, f, indent=2)
    
    def print_status(self):
        """Print current training status to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("TRAINING STATUS DASHBOARD")
        print("="*60)
        
        # Metrics
        metrics = summary['metrics']
        if metrics['avg_loss'] is not None:
            print(f"Average Loss: {metrics['avg_loss']:.4f} (±{metrics['loss_std']:.4f})")
        if metrics['avg_win_rate'] is not None:
            print(f"Average Win Rate: {metrics['avg_win_rate']:.3f}")
        if metrics['avg_game_length'] is not None:
            print(f"Average Game Length: {metrics['avg_game_length']:.1f}")
        if metrics['avg_entropy'] is not None:
            print(f"Average Policy Entropy: {metrics['avg_entropy']:.4f}")
        
        # Health status
        health = summary['health']
        print(f"\nBest Win Rate: {health['best_win_rate']:.3f}")
        print(f"Iterations Without Improvement: {health['iterations_without_improvement']}")
        
        # Recent alerts
        if health['recent_alerts']:
            print("\nRecent Alerts:")
            for alert in health['recent_alerts'][-3:]:  # Show last 3
                print(f"  [{alert['type'].upper()}] {alert['message']}")
        
        print("="*60)


class Connect4InsightTracker:
    """
    Track Connect 4 specific insights during training.
    """
    
    def __init__(self):
        self.opening_moves = deque(maxlen=1000)
        self.win_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
        self.perfect_play_rate = deque(maxlen=100)
        self.move_distribution = np.zeros(7)
        
    def analyze_game(self, game_data: Dict):
        """Analyze a completed Connect 4 game."""
        moves = game_data.get('moves', [])
        outcome = game_data.get('outcome')
        
        if moves:
            # Track opening move
            first_move = moves[0]
            self.opening_moves.append(first_move)
            
            # Check for perfect play (center column opening)
            is_perfect_opening = (first_move == 3)
            self.perfect_play_rate.append(float(is_perfect_opening))
            
            # Update move distribution
            for move in moves:
                if 0 <= move < 7:
                    self.move_distribution[move] += 1
        
    def get_insights(self) -> Dict:
        """Get Connect 4 specific insights."""
        insights = {}
        
        # Opening move analysis
        if self.opening_moves:
            opening_counts = np.bincount(list(self.opening_moves), minlength=7)
            most_common_opening = np.argmax(opening_counts)
            insights['most_common_opening'] = int(most_common_opening)
            insights['center_opening_rate'] = opening_counts[3] / len(self.opening_moves)
        
        # Perfect play rate
        if self.perfect_play_rate:
            insights['recent_perfect_play_rate'] = np.mean(self.perfect_play_rate)
        
        # Move distribution
        if self.move_distribution.sum() > 0:
            normalized_dist = self.move_distribution / self.move_distribution.sum()
            insights['move_distribution'] = normalized_dist.tolist()
            insights['center_preference'] = float(normalized_dist[3])
        
        return insights
    
    def print_insights(self):
        """Print Connect 4 specific insights."""
        insights = self.get_insights()
        
        print("\n" + "-"*40)
        print("CONNECT 4 LEARNING INSIGHTS")
        print("-"*40)
        
        if 'center_opening_rate' in insights:
            print(f"Center Opening Rate: {insights['center_opening_rate']:.1%}")
            print(f"Most Common Opening: Column {insights['most_common_opening']}")
        
        if 'recent_perfect_play_rate' in insights:
            print(f"Recent Perfect Play Rate: {insights['recent_perfect_play_rate']:.1%}")
        
        if 'move_distribution' in insights:
            print("\nMove Distribution:")
            for col, freq in enumerate(insights['move_distribution']):
                bar = "█" * int(freq * 20)
                print(f"  Col {col}: {bar} {freq:.1%}")
        
        print("-"*40)


def create_training_monitor(experiment_name: str = None) -> 'TrainingMonitor':
    """
    Create a comprehensive training monitor combining dashboard and game insights.
    """
    
    class TrainingMonitor:
        def __init__(self, experiment_name: str):
            self.dashboard = TrainingDashboard()
            self.connect4_tracker = Connect4InsightTracker()
            self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        def update(self, metrics: Dict, game_data: Optional[List[Dict]] = None):
            """Update all tracking components."""
            self.dashboard.update_metrics(metrics)
            
            if game_data:
                for game in game_data:
                    self.connect4_tracker.analyze_game(game)
        
        def print_full_status(self):
            """Print comprehensive status update."""
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {self.experiment_name}")
            print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            self.dashboard.print_status()
            self.connect4_tracker.print_insights()
            
        def save_report(self, filename: str = None):
            """Save comprehensive report to file."""
            if filename is None:
                filename = f"report_{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'experiment': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'dashboard_summary': self.dashboard.get_summary(),
                'connect4_insights': self.connect4_tracker.get_insights()
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nReport saved to: {filename}")
    
    return TrainingMonitor(experiment_name)
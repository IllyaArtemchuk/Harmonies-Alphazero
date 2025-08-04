# AlphaZero Training Metrics & Monitoring Guide

## Overview

The enhanced metrics system provides deep insights into whether your AlphaZero model is actually learning. It tracks various metrics during training and provides real-time monitoring through TensorBoard and custom dashboards.

## Key Features

### 1. **TensorBoard Integration**
- Real-time loss curves (total, policy, value)
- Win rate progression graphs
- Learning rate tracking
- Model weight and gradient histograms
- Policy entropy monitoring

### 2. **Training Insights**
- **Loss Monitoring**: Track if losses are decreasing (learning) or exploding
- **Win Rate Tracking**: See actual game performance improvement
- **Policy Entropy**: Monitor exploration vs exploitation balance
- **Value Accuracy**: Check if model correctly evaluates positions

### 3. **Connect 4 Specific Metrics**
- Opening move preferences (should converge to center column)
- Move distribution analysis
- Perfect play rate tracking
- Game length trends

### 4. **Alert System**
- Loss explosion detection
- Training stagnation warnings
- Low entropy alerts (overfitting)
- NaN/Inf value detection

## Usage

### Basic Setup

1. **Run training with metrics enabled** (already configured in main_connect4.py):
```bash
python main_connect4.py
```

2. **View real-time metrics in TensorBoard**:
```bash
tensorboard --logdir ./runs
```
Then open http://localhost:6006 in your browser.

### What to Look For

#### During Early Training (First 10-20 iterations):
- **Losses should decrease**: Total loss dropping from ~1.0 to ~0.1-0.3
- **High policy entropy**: ~0.7-1.0 (exploring different moves)
- **Random win rate**: ~50% against previous versions
- **Varied opening moves**: Trying different columns

#### Signs of Learning (20-100 iterations):
- **Stable, low losses**: Consistent values around 0.1-0.2
- **Improving win rate**: Gradually increasing above 50%
- **Decreasing entropy**: Model becoming more confident (0.3-0.5)
- **Center column preference**: More games starting with column 3

#### Convergence (100+ iterations):
- **Very low losses**: < 0.1
- **High win rate**: 70-90% against older versions
- **Low but non-zero entropy**: ~0.1-0.3 (some exploration)
- **Optimal play patterns**: Consistent center openings

### Reading the Metrics

#### TensorBoard Scalars Tab:
- **Loss/total**: Should decrease and stabilize
- **Loss/policy**: How well model predicts good moves
- **Loss/value**: How well model evaluates positions
- **Evaluation/win_rate**: Performance against best model
- **Policy/avg_entropy**: Exploration level

#### TensorBoard Histograms Tab:
- **Weights**: Should show stable distributions
- **Gradients**: Should not be all zeros or exploding
- **Policy/entropy_distribution**: Shape of exploration

### Troubleshooting

#### Loss Explosion:
- Check learning rate (try reducing by 10x)
- Verify data preprocessing
- Check for NaN in inputs

#### No Improvement:
- Increase MCTS simulations
- Check if replay buffer is too small
- Verify game implementation

#### Low Entropy Too Early:
- Increase exploration (cpuct parameter)
- Add noise to policy during self-play
- Check temperature settings

## Advanced Features

### Custom Dashboard
The system includes a real-time dashboard that prints:
```
============================================================
TRAINING STATUS DASHBOARD
============================================================
Average Loss: 0.1523 (±0.0234)
Average Win Rate: 0.675
Average Game Length: 28.3
Average Policy Entropy: 0.234

Best Win Rate: 0.725
Iterations Without Improvement: 5

Recent Alerts:
  [INFO] Low policy entropy: 0.1823 - model may be overfitting
============================================================

----------------------------------------
CONNECT 4 LEARNING INSIGHTS
----------------------------------------
Center Opening Rate: 78.5%
Most Common Opening: Column 3
Recent Perfect Play Rate: 82.0%

Move Distribution:
  Col 0: ████ 11.2%
  Col 1: ██████ 14.3%
  Col 2: ████████ 18.1%
  Col 3: ████████████████ 35.2%
  Col 4: ████████ 17.8%
  Col 5: ██ 2.1%
  Col 6: █ 1.3%
----------------------------------------
```

### Automated Reports
JSON reports are saved periodically with comprehensive metrics for later analysis.

## Tips for Better Learning

1. **Start with smaller networks** for Connect 4 (4 res blocks, 64 filters)
2. **Use appropriate MCTS simulations** (200 for Connect 4)
3. **Monitor metrics frequently** during first 50 iterations
4. **Adjust hyperparameters** based on metric feedback
5. **Save checkpoints** when win rate improves significantly

## Example Analysis

Run the example to see metrics in action:
```bash
python example_metrics_usage.py
```

This demonstrates how metrics change during successful training.
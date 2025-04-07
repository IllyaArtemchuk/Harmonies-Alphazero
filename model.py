from torch import *


import torch
import torch.nn as nn
import torch.optim as optim
import os
from config import *
from pathlib import Path # Optional, for cleaner path handling

class ModelManager:
    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"Using device: {self.device}")

        # Instantiate the actual neural network model
        self.model = AlphaZeroModel(
            input_channels=INPUT_CHANNELS,
            board_size=BOARD_SIZE,
            action_size=ACTION_SIZE
        ).to(self.device) # Move model to the chosen device

        # Define the optimizer
        self.learning_rate = LEARNING_RATE
        self.optimizer = optim.Adam( # Or optim.SGD, etc.
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=REG_CONST
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.value_loss_weight = VALUE_LOSS_WEIGHT
        self.policy_loss_weight = POLICY_LOSS_WEIGHT

    def predict(self, state_tensor):
        """
        Gets policy and value predictions for a given state tensor.

        Args:
            state_tensor (torch.Tensor): Input tensor for the model 
                                        (should include all required channels/features).

        Returns:
            tuple: (policy_probs (np.ndarray), value (float)) - Detached from graph, on CPU.
        """
        # Ensure tensor is on the correct device and has batch dimension
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0) # Add batch dim if missing
        state_tensor = state_tensor.to(self.device)

        self.model.eval() # Set model to evaluation mode (disables dropout, affects batchnorm)
        with torch.no_grad(): # Disable gradient calculations for inference
            policy_logits, value = self.model(state_tensor) # Assuming forward returns logits for policy
            policy_probs = torch.softmax(policy_logits, dim=1)

        # Detach, move to CPU, convert to numpy
        policy_probs_np = policy_probs.squeeze(0).detach().cpu().numpy()
        value_np = value.squeeze(0).item() # .item() gets scalar from tensor

        return policy_probs_np, value_np


    def train_step(self, states, target_policies, target_values):
        """
        Performs a single training step on a batch of data.

        Args:
            states (torch.Tensor): Batch of input state tensors.
            target_policies (torch.Tensor): Batch of target policy vectors (pi).
            target_values (torch.Tensor): Batch of target values (z).

        Returns:
            tuple: (total_loss, policy_loss, value_loss) - Scalar tensor values.
        """
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device).unsqueeze(1) # Ensure value target has shape (batch, 1)

        self.model.train() # Set model to training mode
        self.optimizer.zero_grad() # Reset gradients

        # Forward pass
        policy_logits, value_pred = self.model(states)

        # Calculate losses
        policy_loss = self.policy_loss_fn(policy_logits, target_policies)
        value_loss = self.value_loss_fn(value_pred, target_values)
        total_loss = (self.policy_loss_weight * policy_loss + 
                      self.value_loss_weight * value_loss)

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()


    def save_checkpoint(self, folder='checkpoints', filename='checkpoint.pth.tar'):
        """ Saves model and optimizer state. """
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        filepath = folder_path / filename

        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Add other things if needed: epoch, best_loss, etc.
        }
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")


    def load_checkpoint(self, folder='checkpoints', filename='checkpoint.pth.tar'):
        """ Loads model and optimizer state. """
        filepath = Path(folder) / filename
        
        if not filepath.exists():
            print(f"WARNING: No checkpoint found at {filepath} - starting model from scratch.")
            return False

        try:
            # Load checkpoint onto the correct device
            checkpoint = torch.load(filepath, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # If you saved other states like epoch, load them here
            # self.start_epoch = checkpoint['epoch'] 

            print(f"Checkpoint loaded from {filepath}")
            return True # Indicate successful load
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint from {filepath}. Error: {e}")
            print("Starting model from scratch.")
            return False


    
class AlphaZeroModel(nn.Module):
    def __init__(self, input_channels=38, board_size=(5, 6), action_size=28):
        super(AlphaZeroModel, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 75, kernel_size=2, padding='same')
        self.bn = nn.BatchNorm2d(75)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(75, kernel_size=2) for _ in range(6)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(75, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size[0] * board_size[1], action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(75, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size[0] * board_size[1], 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Initial convolution
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = torch.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy) # Run through fully-connected layer
        policy = torch.softmax(policy, dim=1)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = torch.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.value_fc1(value)
        value = torch.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)
        
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=2):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += residual
        
        # Final activation
        out = torch.relu(out)
        
        return out
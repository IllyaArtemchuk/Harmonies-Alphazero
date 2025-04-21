import torch
from torch import optim, nn
from pathlib import Path  # Optional, for cleaner path handling
from config_types import TrainingConfigType, ModelConfigType


class ModelManager:
    def __init__(
        self, model_config: ModelConfigType, training_config: TrainingConfigType
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device(training_config["device"])
        print(f"Using device: {self.device}")

        # Instantiate the actual neural network model
        self.model = AlphaZeroModel(
            input_channels=model_config["input_channels"],
            cnn_filters=model_config["cnn_filters"],
            board_size=model_config["board_size"],
            action_size=model_config["action_size"],
            global_feature_size=model_config["global_feature_size"],
            value_hidden_dim=model_config["value_head_hidden_dim"],
            num_res_blocks=model_config["num_res_blocks"],
            # Add policy/value head conv filter counts if they vary
        ).to(self.device)

        self.learning_rate = training_config["learning_rate"]

        if training_config["optimizer_type"] == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=training_config["weight_decay"],
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=training_config["momentum"]
            )

        self.value_loss_fn = nn.MSELoss()
        self.value_loss_weight = training_config["value_loss_weight"]
        self.policy_loss_weight = training_config["policy_loss_weight"]

    def predict(self, board_tensor, global_features_tensor):
        """
        Gets policy and value predictions for a given state tensor.

        Args:
            state_tensor (torch.Tensor): Input tensor for the model
                                        (should include all required channels/features).

        Returns:
            tuple: (policy_probs (np.ndarray), value (float)) - Detached from graph, on CPU.
        """
        # Ensure tensor is on the correct device and has batch dimension
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)  # Add batch dim if missing
        board_tensor = board_tensor.to(self.device)

        if global_features_tensor.dim() == 1:
            global_features_tensor = global_features_tensor.unsqueeze(0)
        global_features_tensor = global_features_tensor.to(self.device)

        self.model.eval()  # Set model to evaluation mode (disables dropout, affects batchnorm)
        with torch.no_grad():  # Disable gradient calculations for inference
            policy_logits, value = self.model(board_tensor, global_features_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1)

        # Detach, move to CPU, convert to numpy
        policy_probs_np = policy_probs.squeeze(0).detach().cpu().numpy()
        value_np = value.squeeze(0).item()  # .item() gets scalar from tensor

        return policy_probs_np, value_np

    def train_step(
        self, board_tensor, global_features_tensor, target_policies, target_values
    ):
        """
        Performs a single training step on a batch of data.

        Args:
            board_tensor, global_features_tensor (torch.Tensor): Batches of input state tensors.
            target_policies (torch.Tensor): Batch of target policy vectors (pi).
            target_values (torch.Tensor): Batch of target values (z).

        Returns:
            tuple: (total_loss, policy_loss, value_loss) - Scalar tensor values.
        """
        board_tensor = board_tensor.to(self.device)
        global_features_tensor = global_features_tensor.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device).unsqueeze(
            1
        )  # Ensure value target has shape (batch, 1)

        self.model.train()  # Set model to training mode
        self.optimizer.zero_grad()  # Reset gradients

        # Forward pass
        policy_logits, value_pred = self.model(board_tensor, global_features_tensor)

        # Calculate losses

        # policy
        log_policy_pred = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_policies * log_policy_pred, dim=1).mean()

        # value
        value_loss = self.value_loss_fn(value_pred, target_values)

        total_loss = (
            self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
        )

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def save_checkpoint(self, folder="checkpoints", filename="checkpoint.pth.tar"):
        """Saves model and optimizer state."""
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        filepath = folder_path / filename

        state = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Add other things if needed: epoch, best_loss, etc.
        }
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder="checkpoints", filename="checkpoint.pth.tar"):
        """Loads model and optimizer state."""
        filepath = Path(folder) / filename

        if not filepath.exists():
            print(
                f"WARNING: No checkpoint found at {filepath} - starting model from scratch."
            )
            return False

        try:
            # Load checkpoint onto the correct device
            checkpoint = torch.load(filepath, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # If you saved other states like epoch, load them here
            # self.start_epoch = checkpoint['epoch']

            print(f"Checkpoint loaded from {filepath}")
            return True  # Indicate successful load
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint from {filepath}. Error: {e}")
            print("Starting model from scratch.")
            return False


class AlphaZeroModel(nn.Module):
    def __init__(
        self,
        input_channels,
        cnn_filters,
        board_size,
        action_size,
        global_feature_size,
        value_hidden_dim,
        num_res_blocks,
        policy_head_conv_filters=2,
        value_head_conv_filters=1,
    ):
        super(AlphaZeroModel, self).__init__()
        height, width = board_size

        # Initial Conv layer
        self.conv = nn.Conv2d(
            input_channels, cnn_filters, kernel_size=2, padding="same"
        )
        self.bn = nn.BatchNorm2d(cnn_filters)

        # Residual tower
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(cnn_filters, kernel_size=2) for _ in range(num_res_blocks)]
        )

        # --- Policy Head Components ---
        self.policy_conv = nn.Conv2d(
            cnn_filters, policy_head_conv_filters, kernel_size=1
        )
        self.policy_bn = nn.BatchNorm2d(policy_head_conv_filters)
        policy_conv_flat_size = policy_head_conv_filters * height * width
        # FC layer now takes flattened conv + global features
        self.policy_fc = nn.Linear(
            policy_conv_flat_size + global_feature_size, action_size
        )

        # --- Value Head Components ---
        self.value_conv = nn.Conv2d(cnn_filters, value_head_conv_filters, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_head_conv_filters)
        value_conv_flat_size = value_head_conv_filters * height * width
        # FC layer 1 now takes flattened conv + global features
        self.value_fc1 = nn.Linear(
            value_conv_flat_size + global_feature_size, value_hidden_dim
        )
        self.value_fc2 = nn.Linear(value_hidden_dim, 1)

    def forward(self, x_board, x_global):
        # x_body shape [Batch, cnn_filters, H, W]
        # x_global shape [Batch, global_feature_size]
        x = self.conv(x_board)
        x = self.bn(x)
        x = torch.relu(x)

        # Residual tower
        for block in self.residual_blocks:
            x = block(x)

        # --- Policy Head Forward ---
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = torch.relu(policy)
        policy_flat = policy.view(policy.size(0), -1)  # Flatten CONV output
        policy_combined = torch.cat(
            (policy_flat, x_global), dim=1
        )  # ADD Global Features
        policy_logits = self.policy_fc(policy_combined)  # Final FC

        # --- Value Head Forward ---
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = torch.relu(value)
        value_flat = value.view(value.size(0), -1)  # Flatten CONV output
        value_combined = torch.cat((value_flat, x_global), dim=1)  # ADD Global Features
        value = self.value_fc1(value_combined)  # FC 1
        value = torch.relu(value)
        value = self.value_fc2(value)  # FC 2
        value = torch.tanh(value)

        return policy_logits, value


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=2):
        super(ResidualBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding="same"
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

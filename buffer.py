import pickle
import torch
from collections import deque
from pathlib import Path


def save_buffer(buffer, folder="buffer", filename="replay_buffer.pkl"):
    """Saves the deque replay buffer to a file using pickle."""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    filepath = folder_path / filename
    try:
        with open(filepath, "wb") as f:
            pickle.dump(buffer, f, pickle.HIGHEST_PROTOCOL)
        print(f"Replay buffer saved successfully to {filepath} ({len(buffer)} items)")
    except Exception as e:
        print(f"Error saving replay buffer to {filepath}: {e}")


def load_buffer(max_size, folder="buffer", filename="replay_buffer.pkl"):
    """Loads a deque replay buffer from a file using pickle."""
    filepath = Path(folder) / filename
    if filepath.exists():
        try:
            with open(filepath, "rb") as f:
                buffer = pickle.load(f)
            # Ensure it's a deque with the correct maxlen
            if not isinstance(buffer, deque):
                print("Warning: Loaded object is not a deque. Converting.")
                buffer = deque(buffer, maxlen=max_size)
            elif buffer.maxlen != max_size:
                print(
                    f"Warning: Loaded buffer maxlen ({buffer.maxlen}) differs from config ({max_size}). Adjusting."
                )
                # Create new deque with correct maxlen from loaded data
                buffer = deque(list(buffer), maxlen=max_size)

            print(
                f"Replay buffer loaded successfully from {filepath} ({len(buffer)} items)"
            )
            return buffer
        except Exception as e:
            print(f"Error loading replay buffer from {filepath}: {e}")
            print("Starting with an empty buffer.")
            return deque(maxlen=max_size)
    else:
        print(f"No buffer file found at {filepath}. Starting with an empty buffer.")
        return deque(maxlen=max_size)


# --- Simple Dataset Wrapper (Optional but recommended for DataLoader) ---
# This helps PyTorch's DataLoader interact cleanly with the deque buffer


class ReplayBufferDataset(torch.utils.data.Dataset):
    def __init__(self, buffer):
        # It's often safer to work on a copy of the buffer's contents
        # at the time the dataset is created, especially if self-play
        # might modify the buffer while training is happening (if parallelized).
        self.data = list(buffer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the tuple (board_tensor, global_tensor, pi_tensor, value_tensor)
        return self.data[idx]

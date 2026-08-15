"""
MLP architecture and training hyperparameters for the baseline IDS.

Kept separate from train_baseline.py so the architecture is easy to tune
without digging through training logic, and so evaluate.py / the
attacker's FGSM code can import the same architecture definition to
reconstruct the model when loading saved weights.
"""

import torch.nn as nn


# --- Architecture ---
HIDDEN_LAYER_SIZES = [128, 64, 32]
DROPOUT_RATE = 0.3

# --- Training ---
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5  # stop if val loss doesn't improve for this many epochs
RANDOM_STATE = 42


class IDS_MLP(nn.Module):
    """
    Simple feedforward network for tabular network-flow classification.

    input_dim  -> number of features (e.g. 78 for our CICIDS2018 subset)
    num_classes -> number of attack classes (e.g. 8)

    Kept as plain Linear/ReLU/Dropout layers (no exotic architecture)
    so it stays fully differentiable end-to-end -- this matters because
    FGSM needs to compute gradients of the loss w.r.t. the input, which
    only works cleanly with a standard differentiable network like this.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_sizes=None, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        hidden_sizes = hidden_sizes or HIDDEN_LAYER_SIZES

        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Final layer outputs raw logits -- softmax is applied inside
        # CrossEntropyLoss during training, and can be applied manually
        # (torch.softmax) at inference time when you need probabilities.
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
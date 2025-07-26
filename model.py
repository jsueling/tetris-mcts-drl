"""
A Residual neural network model for Tetris,
with policy and value heads for reinforcement learning.
It processes input grids with current Tetrominoes to
produce action probabilities and state values.

This implementation is adapted from the following resources:
https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
https://youtu.be/DkNIBBBvcPs?si=ays1iv7E7LePi9We
"""

import torch
from torch import nn
import numpy as np

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)

class A0ResNet(nn.Module):
    """
    Residual tower neural network for Tetris following the AlphaZero implementation
    The model consists of:
    - A single convolutional block to process the state representation (grid + Tetromino)
        - Conv2d with 256 filters, kernel size 3, stride 1
        - Batch normalisation
        - ReLU activation
    - A series of residual blocks to extract features (19 or 39)
    - Two output heads:
        - Policy head: maps the features to logits for each action available
            - Conv2d with 2 channels, kernel size 1, stride 1
            - Batch normalisation
            - ReLU activation
            - Flatten
            - Linear layer to map the flattened features to action logits
        - Value head: maps the features to a single score value
            - Conv2d with 1 channel, kernel size 1, stride 1
            - Batch normalisation
            - ReLU activation
            - Flatten
            - Hidden Linear layer with 256 units
            - ReLU activation
            - Linear layer to map to a scalar value
            - Tanh activation to constrain the output value between -1 and 1
    The model is trained using a combination of cross-entropy loss for the policy head
    and mean squared error loss for the value head.
    """
    def __init__(
        self,
        num_residual_blocks=19,
        num_actions=40,
        num_channels=256,
        updates_per_iteration=20,
        device=DEVICE
    ):
        super(A0ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            # 1 channel for the grid + 7 channels for the one-hot encoded Tetromino
            nn.Conv2d(
                in_channels=8,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[
                A0ResBlock(num_channels=num_channels) for _ in range(num_residual_blocks)
            ],
        )

        # Leela Chess Zero suggest 32 instead of 2 and 1 respectively
        policy_head_out_channels = 32
        value_head_out_channels = 32

        with torch.no_grad():
            # Calculate the output size after the last residual block using a dummy input
            dummy_input = torch.zeros(1, 8, 20, 10)
            dummy_input = self.conv1(dummy_input)
            dummy_output = self.residual_blocks(dummy_input)
            policy_head_output_size = np.prod(dummy_output.shape[2:]) * policy_head_out_channels
            value_head_output_size = np.prod(dummy_output.shape[2:]) * value_head_out_channels

        # Policy head maps the combined features to logits for each action available
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=policy_head_out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(policy_head_out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_head_output_size, num_actions)
        )

        # Value head maps the combined features to a single score value
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=value_head_out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(value_head_out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.device = device
        self.to(device)

        self.optimiser = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=1e-3,
            total_steps=updates_per_iteration,
            div_factor=10,
            anneal_strategy='cos',
            pct_start=0.3
        )

        self.mse_loss = nn.MSELoss()

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ResNet model. Returns policy logits and value prediction
        for a given state (grid + tetromino).
        """
        x = self.conv1(state)
        x = self.residual_blocks(x)
        policy_prediction = self.policy_head(x)
        value_prediction = self.value_head(x)
        return policy_prediction, value_prediction

    def loss(
        self,
        states,
        tree_policies,
        ground_truth_values,
        legal_action_masks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the model. This consists of cross-entropy loss
        between the tree policies and the predicted action probabilities,
        and mean squared error loss between the predicted values and ground truth values.

        Returns:
        - policy_loss: Cross-entropy loss for the policy head
        - value_loss: Mean squared error loss for the value head
        """
        # Forward pass through the model to get predicted action logits and values
        predicted_action_logits, predicted_values = self.forward(states)

        # The value loss is the mean squared error between predicted values and ground truth values
        value_loss = self.mse_loss(predicted_values.squeeze(-1), ground_truth_values)

        # == Handling terminal states in the batch ==
        # If all actions are illegal (terminal),
        # the model should not predict any action

        # Select non-terminal samples that have at least one legal action
        non_terminal_mask = torch.any(legal_action_masks, dim=1)

        # default policy loss
        policy_loss = torch.tensor(0.0, device=self.device)

        # If non-terminal samples exist in the batch
        if non_terminal_mask.any():

            # Filter in non-terminal
            logits_nt = predicted_action_logits[non_terminal_mask]
            policies_nt = tree_policies[non_terminal_mask]
            masks_nt = legal_action_masks[non_terminal_mask]

            # Apply legal action masks to the logits before softmax
            # to ensure only legal actions are considered
            logits_nt[~masks_nt] = -1e9

            # Compute the log probabilities of the predicted actions
            predicted_log_probs_nt = torch.log_softmax(logits_nt, dim=1)

            # Negative log likelihood which penalises the model
            # for predicting actions that differ from the tree policies
            policy_loss = -(policies_nt * predicted_log_probs_nt).sum(dim=1).mean()

        return policy_loss, value_loss

    def save(self, file_path):
        """Save the model state using the specified file path."""
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        """
        Load the model state from the specified file path.
        Returns True if the model was successfully loaded and False otherwise.
        """
        try:
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            return True
        except FileNotFoundError:
            print(f"No model state found at {file_path}")
            return False

class A0ResBlock(nn.Module):
    """
    A residual block for the A0ResNet model.
    Each residual block consists of:
    - Conv2d with 256 channels, kernel size 3, stride 1
    - Batch normalization
    - ReLU activation
    - Conv2d with 256 channels, kernel size 3, stride 1
    - Batch normalization
    - The skip connection that adds the original input to the output of the second convolution
    - ReLU activation
    """
    def __init__(self, num_channels=256):
        super(A0ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through a residual block"""
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet18(A0ResNet):
    """Residual neural network for Tetris based on the ResNet-18 architecture."""
    def __init__(self, layers, num_actions, num_channels=64, device=DEVICE):
        super(ResNet18, self).__init__()
        self.input_channels = num_channels
        self.conv1 = nn.Sequential(
            # 1 channel for the grid + 7 channels for the one-hot encoded Tetromino
            nn.Conv2d(
                in_channels=8,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.residual_blocks = nn.Sequential(
            self._make_layer(ResBlock, num_channels, layers[0], stride=1),
            self._make_layer(ResBlock, num_channels * 2, layers[1], stride=2),
            self._make_layer(ResBlock, num_channels * 4, layers[2], stride=2),
            self._make_layer(ResBlock, num_channels * 8, layers[3], stride=1),
        )

        # Number of spatial features after the last residual block
        count_spatial_features = num_channels * 8

        policy_head_out_channels = 2
        value_head_out_channels = 1

        with torch.no_grad():
            # Calculate the output size after the last residual block using a dummy input
            dummy_input = torch.zeros(1, 8, 20, 10)
            dummy_input = self.conv1(dummy_input)
            dummy_output = self.residual_blocks(dummy_input)
            policy_head_output_size = np.prod(dummy_output.shape[2:]) * policy_head_out_channels
            value_head_output_size = np.prod(dummy_output.shape[2:]) * value_head_out_channels

        # Policy head maps the combined features to logits for each action available
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=count_spatial_features,
                out_channels=policy_head_out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(policy_head_out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_head_output_size, num_actions)
        )

        # Value head maps the combined features to a single score value
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=count_spatial_features,
                out_channels=value_head_out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(value_head_out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def _make_layer(self, block, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(output_channels),
            )
        layers = []
        layers.append(block(self.input_channels, output_channels, stride, downsample))
        # The first layer matches the downsample in how it changes the number of channels
        # and the spatial dimensions (stride). This is done so that the residual shape matches
        # the output shape of the block.
        self.input_channels = output_channels
        for _ in range(1, blocks):
            layers.append(block(self.input_channels, output_channels))

        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    """A residual block for the ResNet model."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through a residual block"""
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

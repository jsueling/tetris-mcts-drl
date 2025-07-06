"""
A Residual neural network model for Tetris,
with policy and value heads for reinforcement learning.
It processes input grids with current Tetrominoes to
produce action probabilities and state values.
"""

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet(nn.Module):
    """Residual neural network for Tetris."""
    def __init__(self, block, layers, num_actions, num_channels=64, device=DEVICE):
        super(ResNet, self).__init__()
        self.input_channels = num_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.residual_blocks = nn.Sequential(
            self._make_layer(block, num_channels, layers[0], stride=1),
            self._make_layer(block, num_channels * 2, layers[1], stride=2),
            self._make_layer(block, num_channels * 4, layers[2], stride=2),
            self._make_layer(block, num_channels * 8, layers[3], stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Number of spatial features after the last residual block
        count_spatial_features = num_channels * 8
        # One-hot encoded feature vector (the number of different tetrominoes)
        count_tetromino_features = 7
        total_features = count_spatial_features + count_tetromino_features

        # Policy head maps the combined features to logits for each action available
        self.policy_head = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        # Value head maps the combined features to a single score value
        self.value_head = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.device = device
        self.to(device)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            patience=3,
            factor=0.5
        )

        self.mse_loss = nn.MSELoss()

    def _make_layer(self, block, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels),
            )
        layers = []
        layers.append(block(self.input_channels, output_channels, stride, downsample))
        self.input_channels = output_channels
        for _ in range(1, blocks):
            layers.append(block(self.input_channels, output_channels))

        return nn.Sequential(*layers)

    def forward(self, grid, tetromino):
        """
        Forward pass through the ResNet model. Returns policy logits and value prediction
        for a given state (grid + tetromino).
        """

        x = self.conv1(grid)
        x = self.residual_blocks(x)
        x = self.avgpool(x)

        grid_features = torch.flatten(x, 1)
        combined_features = torch.cat([grid_features, tetromino], dim=1)

        policy_prediction = self.policy_head(combined_features)
        value_prediction = self.value_head(combined_features)

        return policy_prediction, value_prediction

    def loss(
            self,
            grids,
            tetrominoes_one_hot,
            tree_policies,
            legal_action_masks,
            ground_truth_values
        ):
        """
        Compute the loss for the model. This consists of cross-entropy loss
        between the tree policies and the predicted action probabilities,
        and mean squared error loss between the predicted values and ground truth values.
        """
        # Forward pass through the model to get predicted action logits and values
        predicted_action_logits, predicted_values = self.forward(grids, tetrominoes_one_hot)
        # Apply legal action masks to the predicted action logits before softmax
        mask_legal = legal_action_masks == 0
        predicted_action_logits[mask_legal] = -torch.inf
        # Compute the log probabilities of the predicted actions
        predicted_log_probs = torch.log_softmax(predicted_action_logits, dim=1)
        # Negative log likelihood which penalises the model
        # for predicting actions that differ from the tree policies
        policy_loss = -(tree_policies * predicted_log_probs).sum(dim=1).mean()
        # The value loss is the mean squared error between predicted values and ground truth values
        value_loss = self.mse_loss(predicted_values.squeeze(-1), ground_truth_values)
        return policy_loss + value_loss

class ResidualBlock(nn.Module):
    """A residual block for the ResNet model."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
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

"""
A Residual neural network model for Tetris,
with policy and value heads for reinforcement learning.
It processes input grids with current Tetrominoes to
produce action probabilities and state values.
"""

import torch
from torch import nn

class ResNet(nn.Module):
    """Residual neural network for Tetris."""
    def __init__(self, block, layers, num_actions, num_channels=64):
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

    def forward(self, grid, tetromino, training=False):
        """Forward pass through the ResNet model."""

        if training:
            self.train()
        else:
            self.eval()

        x = self.conv1(grid)
        x = self.residual_blocks(x)
        x = self.avgpool(x)

        grid_features = torch.flatten(x, 1)
        combined_features = torch.cat([grid_features, tetromino], dim=1)

        p, v = self.policy_head(combined_features), self.value_head(combined_features)

        return p, v

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

# ACTION_SPACE = 40
# # ResNet-18
# model = ResNet(ResidualBlock, [2, 2, 2, 2], ACTION_SPACE)
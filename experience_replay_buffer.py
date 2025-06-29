"""Experience replay buffer for storing and sampling transitions."""

import random

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

class ExperienceReplayBuffer:
    """An experience replay buffer that stores transitions and supports uniform random sampling."""
    def __init__(
        self,
        max_size: int = 1000000,
        batch_size: int = BATCH_SIZE,
        device: torch.device = DEVICE,
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.buffer = []
        self.position = 0

        # The buffer will store transitions in the form of tuples:
        # 1. State will be (grid + piece type)
        # 2. Tree policy derived from MCTS associated with the state
        # 3. Value for the state calculated as
        # reward to go when an actual game ends (end_score - current_score)

    def add_transition(self, transition):
        """Add a single transition to the buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.max_size

    def add_transitions_batch(self, transitions):
        """Add batch of transitions to the buffer."""
        for transition in transitions:
            self.add_transition(transition)

    def sample(self):
        """Sample a random batch of transitions uniformly from the buffer."""
        return random.sample(self.buffer, min(self.batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

"""Script for running the training of the MCTS agent."""

import random
import numpy as np
import torch

from mcts_agent import MCTSAgent

if __name__ == "__main__":

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    agent = MCTSAgent()
    agent.train()

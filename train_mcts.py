"""Script for running the training of the MCTS agent."""

import multiprocessing as mp
import asyncio

import random
import numpy as np
import torch

from mcts_agent import MCTSAgent

if __name__ == "__main__":

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        # Benchmarks, then caches most efficient convolution algorithms
        # given the current configuration. Do not use if input sizes change frequently
        torch.backends.cudnn.benchmark = True

    agent = MCTSAgent()
    # agent.train_ensemble()
    asyncio.run(agent.train_async())

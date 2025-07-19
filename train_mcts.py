"""Script for running the training of the MCTS agent."""

import multiprocessing as mp
import argparse

import random
import numpy as np
import torch

from mcts_agent_async import MCTSAgentAsync
# from mcts_agent_ensemble import MCTSAgentEnsemble

if __name__ == "__main__":

    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        # Benchmarks, then caches most efficient convolution algorithms
        # given the current configuration. Do not use if input sizes change frequently
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Train MCTS agent to play Tetris.")

    parser.add_argument(
        "--checkpoint_name",
        "-c",
        required=True,
        type=str,
        help="Name of the checkpoint for loading and saving the state of training"
    )

    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    checkpoint_name = args.checkpoint_name + f"_seed_{args.seed}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = MCTSAgentAsync(checkpoint_name=checkpoint_name)
    # agent = MCTSAgentEnsemble(checkpoint_name=checkpoint_name)
    agent.train()

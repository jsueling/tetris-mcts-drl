"""Script for starting the training of a Deep MCTS agent that learns to play Tetris."""

import multiprocessing as mp
import argparse

import random
import numpy as np
import torch

from deep_mcts_agent_async import DeepMCTSAgentAsync
from deep_mcts_agent_ensemble import DeepMCTSAgentEnsemble

if __name__ == "__main__":

    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        # Benchmarks, then caches most efficient convolution algorithms
        # given the current configuration. Do not use if input sizes change frequently
        torch.backends.cudnn.benchmark = True
        # Enables use of fast TF32 Tensor Cores for matrix multiplications
        torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="Train Deep MCTS agent to play Tetris.")

    parser.add_argument(
        "--agent_type",
        "-a",
        choices=["async", "ensemble"],
        required=True,
        help="Type of Deep MCTS agent to train: 'async' for DeepMCTSAgentAsync or "
             "'ensemble' for DeepMCTSAgentEnsemble."
    )

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

    checkpoint_name = args.checkpoint_name + f"_{args.agent_type}_seed_{args.seed}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.agent_type == "async":
        agent = DeepMCTSAgentAsync(checkpoint_name=checkpoint_name)
        agent.train()
    elif args.agent_type == "ensemble":
        agent = None
        try:
            agent = DeepMCTSAgentEnsemble(checkpoint_name=checkpoint_name)
            agent.train()
        finally:
            # Clean up resources
            if agent is not None:
                agent.stop()

"""MCTS + DRL agent that learns how to play Tetris"""

import torch
import numpy as np
from tqdm import tqdm

from mcts import MonteCarloTreeNode
from experience_replay_buffer import ExperienceReplayBuffer
from tetris_env import Tetris
from model import ResNet, ResidualBlock

MCTS_ITERATIONS = 800 # Number of MCTS iterations per action selection
ACTION_SPACE = 40 # Upper bound on possible actions for hard drop (rotations * placements)
BATCH_SIZE = 128 # Batch size for experience replay

class MCTSAgent:
    """MCTS + DRL agent for playing Tetris."""
    def __init__(self, batch_size=BATCH_SIZE):
        self.model = ResNet(ResidualBlock, [2, 2, 2, 2], ACTION_SPACE)
        self.buffer = ExperienceReplayBuffer(
            batch_size=batch_size,
            max_size=1e6,
            device=self.model.device
        )
        self.batch_size = batch_size
        self.env = Tetris()

    def update(self):
        """Update the agent's model via experience replay"""

        xp_batch = self.buffer.sample()
        device = self.model.device

        # (grids, tetromino_types), tree_policies, values = zip(*batch)
        states, tree_policies, actual_rewards = zip(*xp_batch)
        grids, tetromino_types = zip(*states)
        grids = torch.tensor(grids, dtype=torch.float32).to(device)
        tetrominoes_one_hot = torch.zeros((self.batch_size, 7), dtype=torch.float32).to(device)
        tetrominoes_one_hot[torch.arange(self.batch_size), tetromino_types] = 1.0
        tree_policies = torch.tensor(tree_policies, dtype=torch.float32).to(device)
        actual_rewards = torch.tensor(actual_rewards, dtype=torch.float32).to(device)

        legal_actions_masks = []
        temp_env = Tetris()
        for grid, tetromino_type in zip(grids, tetromino_types):
            np.copyto(temp_env.grid, grid)
            temp_env.create_tetromino(tetromino_type)
            mask = temp_env.get_legal_actions()
            legal_actions_masks.append(mask)
        legal_actions_masks = np.stack(legal_actions_masks)

        self.model.train()
        self.model.optimiser.zero_grad()
        loss = self.model.loss(
            grids,
            tetrominoes_one_hot,
            tree_policies,
            legal_actions_masks,
            actual_rewards
        )
        loss.backward()
        self.model.optimiser.step()

    def train(self, episodes=1000):
        """Run the training loop for the MCTS agent."""

        step_count = 0
        scores = []

        for episode in range(episodes):

            # Create a new root node for the MCTS search tree per episode.
            # Since Tetromino generation is stochastic, the old tree must be discarded.
            root_node = MonteCarloTreeNode(env=self.env, model=self.model)

            # Reset the environment per episode, generating the first Tetromino of the sequence
            self.env.reset()
            transitions = []
            done = False
            while not done:

                state = self.env.get_state()

                for mcts_iteration in tqdm(
                    range(MCTS_ITERATIONS),
                    desc=f"{episode}, {mcts_iteration}"
                ):
                    root_node.run_iteration()

                # The next action is decided based on the visit counts of the actions
                # available to the root node in the simulations. The tree policy is the
                # probability distribution over those actions
                action, tree_policy = root_node.decide_action(tau=1.0)

                done, current_score = self.env.step(action)

                # Actual environment randomly generates the next Tetromino
                self.env.create_tetromino(self.env.generate_next_tetromino_type())

                transitions.append([state, tree_policy, current_score])

                if (step_count > 1e5) and (step_count % self.batch_size == 0):
                    self.update()

                step_count += 1

            # An episode has ended, we can now compute the return-to-go (RTG)
            # by subtracting the final score from each transition's current total reward (score)
            final_score = self.env.score
            scores.append(final_score)
            if (episode + 1) % 10 == 0:
                avg_score = sum(scores[-10:]) / 10
                print(f"Episode {episode+1}, Average Score (last 10): {avg_score}")
            for t in transitions:
                t[2] = final_score - t[2]

            self.buffer.add_transitions_batch(transitions)

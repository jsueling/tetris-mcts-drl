"""MCTS + DRL agent that learns how to play Tetris"""

import os
import time

import numpy as np
from tqdm import tqdm

from mcts import MonteCarloTreeNode
from experience_replay_buffer import ExperienceReplayBuffer
from tetris_env import Tetris
from model import ResNet, ResidualBlock

MCTS_ITERATIONS = 800 # Number of MCTS iterations per action selection
ACTION_SPACE = 40 # Upper bound on possible actions for hard drop (rotations * columns placements)
BATCH_SIZE = 256 # Batch size for experience replay

class MCTSAgent:
    """MCTS + DRL agent for playing Tetris."""
    def __init__(self, batch_size=BATCH_SIZE):
        self.model = ResNet(ResidualBlock, [2, 2, 2, 2], ACTION_SPACE)
        self.buffer = ExperienceReplayBuffer(
            batch_size=batch_size,
            max_size=1000000,
            device=self.model.device
        )
        self.batch_size = batch_size
        self.env = Tetris()

    def update(self):
        """Update the agent's model via experience replay"""

        grids, tetrominoes_one_hot, tree_policies, \
            rewards_to_go, legal_actions_masks = self.buffer.sample()

        self.model.train()
        self.model.optimiser.zero_grad()
        loss = self.model.loss(
            grids,
            tetrominoes_one_hot,
            tree_policies,
            rewards_to_go,
            legal_actions_masks
        )
        loss.backward()
        self.model.optimiser.step()

    def train(self, episodes=10000):
        """Run the training loop for the MCTS agent."""

        os.makedirs("./out", exist_ok=True)

        prev_step_count = step_count = 0
        episode_scores = []
        rolling_avg_scores = []
        steps_per_episode = []
        episode_times = []

        for _ in tqdm(range(episodes)):

            exploration_steps = 0

            start_time = time.time()

            # Reset the environment per episode, generating the first Tetromino of the sequence
            self.env.reset()
            transitions = []
            done = False
            while not done:

                # Since Tetromino generation is stochastic, the old tree must be discarded
                # after each step in the actual environment.
                root_node = MonteCarloTreeNode(
                    env=self.env.copy(),
                    model=self.model,
                    is_root=True,
                )

                grid, tetromino_one_hot = self.env.get_state()

                for _ in range(MCTS_ITERATIONS):
                    root_node.run_iteration()

                # The next action is decided based on the visit counts of the actions
                # available to the root node in the simulations. The tree policy is the
                # probability distribution over those actions
                action, tree_policy = root_node.decide_action(
                    # First 30 steps of the episode are exploratory,
                    # then the agent only exploits
                    tau=(1.0 if exploration_steps < 30 else 0.0)
                )

                # If the action is -1, no legal actions are available
                if action == -1:
                    done, current_score = True, self.env.score
                else:
                    done, current_score = self.env.step(action)

                # Actual environment randomly generates the next Tetromino
                self.env.create_tetromino(self.env.generate_next_tetromino_type())

                legal_actions = np.zeros(ACTION_SPACE, dtype=np.float32)
                # Legal actions were determined by the children of the root node
                legal_actions[list(root_node.children.keys())] = 1.0

                transitions.append([
                    grid,
                    tetromino_one_hot,
                    tree_policy,
                    current_score,
                    legal_actions
                ])

                if all([
                    # Allows for sufficient diversity in transitions before sampling
                    step_count >= 1e3,
                    # 1:1 ratio of data generated-to-consumed means that
                    # each transition is expected to be sampled for training
                    # once before being replaced on average.
                    step_count % self.batch_size == 0
                ]):
                    self.update()

                step_count += 1
                exploration_steps += 1

            final_score = self.env.score

            episode_scores.append(final_score)
            rolling_avg_scores.append(np.mean(episode_scores[-100:]))
            steps_per_episode.append(step_count - prev_step_count)
            end_time = time.time()
            episode_times.append(end_time - start_time)

            results = {
                "episode_scores": episode_scores,
                "rolling_avg_scores": rolling_avg_scores,
                "steps_per_episode": steps_per_episode,
                "episode_times": episode_times,
                "total_steps": step_count,
                "total_episodes": len(episode_scores),
            }

            np.save("./out/tetris_mcts_drl_results.npy", results)

            prev_step_count = step_count

            # After each episode compute the return-to-go (RTG)
            # by subtracting the final score from each transition's current total reward (score)
            for t in transitions:
                t[3] = final_score - t[3]

            self.buffer.add_transitions_batch(transitions)

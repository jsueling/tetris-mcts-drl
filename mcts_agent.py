"""MCTS + DRL agent that learns how to play Tetris"""

import os
import time
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from mcts import MonteCarloTreeNode
from experience_replay_buffer import ExperienceReplayBuffer
from tetris_env import Tetris
from model import ResNet, ResidualBlock
from score_normaliser import ScoreNormaliser
from inference_server import InferenceServer

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
        self.score_normaliser = ScoreNormaliser()

    def update(self):
        """Update the agent's model via experience replay"""

        states, tree_policies, rewards_to_go, legal_actions_masks = self.buffer.sample()

        self.model.train()
        self.model.optimiser.zero_grad()
        loss = self.model.loss(
            states,
            tree_policies,
            rewards_to_go,
            legal_actions_masks
        )
        loss.backward()
        self.model.optimiser.step()

    def train(self, episodes=10000, num_workers=10):
        """Run the training loop for the MCTS agent."""

        os.makedirs("./out", exist_ok=True)

        prev_step_count = step_count = 0
        episode_scores = []
        rolling_avg_scores = []
        steps_per_episode = []
        episode_times = []

        # Pre-allocate queues for parallel MCTS simulations. Each worker has its own response queue
        # to avoid contention when receiving responses from the queue.
        response_queues = { worker_id: mp.Queue() for worker_id in range(num_workers) }
        request_queue = mp.Queue()
        result_queue = mp.Queue()

        # Create and start the inference server which handles model inference
        # requests sent by the MCTS workers.
        inference_server = InferenceServer(
            self.model,
            request_queue=request_queue,
            response_queues=response_queues
        )
        inference_server.start()

        for _ in tqdm(range(episodes)):

            exploration_steps = 0

            start_time = time.time()

            # Reset the environment per episode, generating the first Tetromino of the sequence
            self.env.reset()
            transitions = []
            rewards_to_go = []
            done = False
            while not done:

                # Since Tetromino generation is stochastic, the old tree must be discarded
                # after each step in the actual environment.

                # Multiple workers run MCTS iterations in parallel from the root.
                # An average of the resulting visit counts is taken to account
                # for stochasticity in Tetromino generation. Each tree search
                # has a single determinised future (Ensemble UCT)

                processes = []

                for worker_id in range(num_workers):
                    p = mp.Process(
                        target=mcts_simulation_helper,
                        args=(
                            self.env.copy(),
                            request_queue,
                            response_queues[worker_id],
                            worker_id,
                            result_queue,
                            MCTS_ITERATIONS
                        )
                    )
                    p.start()
                    processes.append(p)

                # Block until all processes are finished
                for p in processes:
                    p.join()

                # Fetch results from all processes
                root_nodes = [result_queue.get() for _ in processes]

                # Stack visit counts from all root nodes and compute the mean
                mean_visit_counts = np.mean(
                    [root_node["visit_counts"] for root_node in root_nodes],
                    axis=0
                )

                # The next action is decided based on the visit counts of the actions
                # available to the root node in the simulations. The tree policy is the
                # probability distribution over those actions

                tree_policy = np.zeros(ACTION_SPACE, dtype=np.float32)

                # All actions available to the root node (same for any of the root nodes)
                actions = root_nodes[0]["actions"]

                if len(actions) == 0:
                    action = -1
                elif exploration_steps < 30:
                    action = actions[np.argmax(mean_visit_counts)]
                    tree_policy[action] = 1.0
                else:
                    # The temperature (tau) is set to 1 after the first 30 steps which has no effect
                    # i.e. visit_counts ** (1 / tau)
                    visit_probabilities = mean_visit_counts / np.sum(mean_visit_counts)
                    tree_policy[actions] = visit_probabilities
                    action = np.random.choice(ACTION_SPACE, p=tree_policy)

                state = self.env.get_state()

                # If the action is -1, no legal actions are available
                if action == -1:
                    done, current_score = True, self.env.score
                else:
                    done, current_score = self.env.step(action)

                # Actual environment randomly generates the next Tetromino
                self.env.create_tetromino(self.env.generate_next_tetromino_type())

                legal_actions = np.zeros(ACTION_SPACE, dtype=np.float32)
                # Legal actions were determined by the children of the root node
                legal_actions[actions] = 1.0

                transitions.append([
                    state,
                    tree_policy,
                    legal_actions
                ])
                rewards_to_go.append(current_score)

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

            np.save("./out/tetris_mcts_drl_results.npy", np.array(results))

            prev_step_count = step_count

            # After each episode compute the return-to-go (RTG)
            # then normalise it to [-1, 1] range based on rolling average score
            # since Tetris scores are unbounded. This should encourage the agent to
            # learn to play better than its current iteration rather than just score higher.
            # When used in MCTS node selection, the assumption is that moves that were improvements
            # on average score in the past will also be improvements on average score in the future
            # which is a reasonable assumption for Tetris due to the repeated nature of the game.
            for i, transition in enumerate(transitions):
                rewards_to_go[i] = final_score - rewards_to_go[i]
                normalised_rtg = self.score_normaliser.normalise(rewards_to_go[i])
                transition.append(normalised_rtg)

            self.score_normaliser.update(rewards_to_go)
            self.buffer.add_transitions_batch(transitions)

        inference_server.stop()

def mcts_simulation_helper(
        env: Tetris,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int,
        result_queue: mp.Queue,
        iterations: int,
    ) -> MonteCarloTreeNode:
    """
    Helper function used by a single process for parallel simulations that runs MCTS
    iterations on a single node and puts the results into the result queue.
    """

    root_node = MonteCarloTreeNode(
        env=env,
        is_root=True,
        request_queue=request_queue,
        response_queue=response_queue,
        worker_id=worker_id
    )

    for _ in range(iterations):
        root_node.run_iteration()

    result_queue.put({
        "actions": list(root_node.children.keys()),
        "visit_counts": [child.visit_count for child in root_node.children.values()]
    })

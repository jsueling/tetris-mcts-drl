"""
Deep MCTS agent ensemble for Tetris which uses parallelised workers to explore
isolated, determinised futures. It takes the majority vote on action
selection and tree policy derived from visit counts.
"""

import random
import zlib

import torch
from torch import multiprocessing as torch_mp
import numpy as np

from deep_mcts_agent import DeepMCTSAgent, ACTION_SPACE, MCTS_ITERATIONS, BATCH_SIZE
from tetris_env import Tetris
from mcts import MCTreeNodeDeterminised
from inference_server import InferenceServer
from model import A0ResNet

RANDOM_SEED_MODULUS = 2 ** 32 # Seed methods accept 32-bit integers only

class DeepMCTSAgentEnsemble(DeepMCTSAgent):
    """Deep MCTS agent ensemble that learns how to play Tetris."""

    def __init__(self, checkpoint_name, batch_size=BATCH_SIZE):
        super(DeepMCTSAgentEnsemble, self).__init__(checkpoint_name, batch_size=batch_size)

        # Queues for inter-process communication

        # Sends tasks to workers
        self.task_queue = torch_mp.Queue()
        # Receives results from workers
        self.result_queue = torch_mp.Queue()
        # Queues for inference requests and responses within process
        self.request_queue = torch_mp.Queue()
        self.response_queues = {
            worker_id: torch_mp.Queue() for worker_id in range(self.n_workers)
        }

        # Create and start the synchronous inference server which handles model inference
        # requests sent by the MCTS workers.
        self.inference_server = InferenceServer(
            model_base_class=A0ResNet,
            model_state_dict=self.model.state_dict(),
            model_init_params={
                "num_residual_blocks": 19,
                "updates_per_iteration": self.updates_per_iteration,
                "num_actions": ACTION_SPACE
            },
            request_queue=self.request_queue,
            response_queues=self.response_queues,
            n_workers=self.n_workers
        )
        self.inference_server.start()
        self.episode_count = 0

        # Shared memory to reduce overhead from inter-process communication
        state_shape = (8, 20, 10)
        self.shared_buffers = [
            torch.zeros(state_shape, dtype=torch.float32)
            for _ in range(self.n_workers)
        ]
        for buffer in self.shared_buffers:
            buffer.share_memory_()

        self.worker_processes = []
        for worker_id in range(self.n_workers):
            p = torch_mp.Process(
                target=ensemble_mcts_helper,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.request_queue,
                    self.response_queues[worker_id],
                    worker_id,
                    self.shared_buffers[worker_id],
                    MCTS_ITERATIONS
                )
            )
            p.start()
            self.worker_processes.append(p)

    def stop(self):
        """Stops the inference server and all worker processes."""
        # None is the sentinel value to signal workers to stop
        for _ in range(self.n_workers):
            self.task_queue.put(None)
        for p in self.worker_processes:
            p.join()
        self.inference_server.stop()
        self.task_queue.close()
        self.result_queue.close()
        self.task_queue.join_thread()
        self.result_queue.join_thread()

    def run_episode(self, model, benchmark=False):
        """
        Runs a single episode of a Deep MCTS agent ensemble that explores separate
        determinised futures in parallel, taking the majority vote on action selection
        and tree policy for model updates.
        """

        self.env.reset()

        self.inference_server.set_model(model)

        # Since each worker runs its own separate determinised future, the trees cannot
        # be reused across episodes and are thus discarded.

        transitions = []
        done = False
        step_count = 0
        while not done:

            # Take the majority vote of the MCTS ensemble on which action to take and
            # the tree policy based on aggregate visit counts.
            chosen_action, available_actions, tree_policy =  \
                self.run_ensemble_mcts(
                    step_count,
                    benchmark=benchmark
                )

            if benchmark is False:

                score_before_action = self.env.score
                state_before_action = self.env.get_state()
                legal_actions = np.zeros(ACTION_SPACE, dtype=np.float32)
                legal_actions[available_actions] = 1.0

                transitions.append([
                    state_before_action,
                    tree_policy,
                    legal_actions,
                    score_before_action
                ])

            # Update the environment with the selected action and next Tetromino
            done = self.env.step(chosen_action)
            self.env.create_tetromino(self.env.generate_next_tetromino_type())

            step_count += 1

        final_score = self.env.score

        self.episode_count += 1

        if benchmark is True:
            return final_score, None, None

        return final_score, transitions, step_count

    def run_ensemble_mcts(
        self,
        step_count: int,
        explore_threshold=30,
        tau=1.0, # Temperature parameter for action selection
        benchmark=False
    ):
        """
        Multiple workers run MCTS iterations in parallel from the root node.
        The stochastic outcome (Tetromino generation) is determinised per
        node expansion i.e. a single future is sampled and shared among child nodes.
        Therefore, we take the average/majority vote from many parallel determinisations
        to get a more robust action decision/transitions at the root.

        Returns:
        - chosen_action: The action selected by the MCTS ensemble
        - available_actions: Actions available to the root node
        - tree_policy: The probability distribution over actions given by mean visit counts
        """

        for worker_id in range(self.n_workers):

            # Each process has its own seed for reproducibility and to prevent the same
            # random number generation
            seed_tuple = (self.episode_count, step_count, worker_id)
            seed_string = "-".join(map(str, seed_tuple))
            process_seed = zlib.adler32(seed_string.encode('utf-8')) % RANDOM_SEED_MODULUS

            task = {
                "env_state": self.env.save_partial_state(),
                "process_seed": process_seed
            }
            self.task_queue.put(task)

        # Fetch results from all processes
        results = [self.result_queue.get() for _ in range(self.n_workers)]

        # Stack visit counts from all root nodes and compute the mean
        mean_visit_counts = np.mean([res["visit_counts"] for res in results], axis=0)
        # Available actions are the same for all workers at the root node
        available_actions = results[0]["actions"]

        # The next action and tree policy is derived from the aggregate visit counts
        # of actions available to the root nodes in the simulations.

        tree_policy = np.zeros(ACTION_SPACE, dtype=np.float32)

        if len(available_actions) == 0:
            # Game ends
            chosen_action = -1
        elif (benchmark is True) or (step_count >= explore_threshold):
            # Greedy when benchmarking or after exploratory steps
            highest_visit_actions = \
                available_actions[mean_visit_counts == np.max(mean_visit_counts)]
            chosen_action = np.random.choice(highest_visit_actions)
            tree_policy[chosen_action] = 1.0
        else:
            # Allow exploratory steps at episode start
            mean_visit_counts = mean_visit_counts ** (1 / tau)
            visit_probabilities = mean_visit_counts / np.sum(mean_visit_counts)
            tree_policy[available_actions] = visit_probabilities
            chosen_action = np.random.choice(ACTION_SPACE, p=tree_policy)

        return chosen_action, available_actions, tree_policy

def ensemble_mcts_helper(
    task_queue: torch_mp.Queue,
    result_queue: torch_mp.Queue,
    request_queue: torch_mp.Queue,
    response_queue: torch_mp.Queue,
    worker_id: int,
    shared_buffer: torch.Tensor,
    iterations: int
):
    """
    Helper function used by a single process for parallel simulations that runs MCTS
    iterations on a single node and puts the results into the result queue.
    """

    env = Tetris()

    while True:

        task = task_queue.get()

        if task is None:
            break

        env.load_partial_state(task["env_state"])
        process_seed = task["process_seed"]

        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        random.seed(process_seed)

        root_node = MCTreeNodeDeterminised(
            env=env,
            request_queue=request_queue,
            response_queue=response_queue,
            worker_id=worker_id,
            shared_buffer=shared_buffer,
            is_root=True
        )

        for _ in range(iterations):
            root_node.run_iteration()

        # A terminal node is never expanded, so has no available actions.
        if root_node.available_actions is None:
            actions = np.array([], dtype=np.int32)
            visit_counts = np.array([], dtype=np.float32)
        else:
            actions = root_node.available_actions
            visit_counts = root_node.visit_counts[actions]

        result_queue.put({
            "actions": actions,
            "visit_counts": visit_counts,
        })

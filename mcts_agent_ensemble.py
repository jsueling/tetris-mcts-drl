"""
MCTS DRL agent ensemble for Tetris using parallelised workers to explore
isolated, determinised futures and take the majority vote on action
selection and tree policy derived from visit counts.
"""

import multiprocessing as mp
import random

import torch
import numpy as np

from mcts_agent import MCTSAgent, ACTION_SPACE, MCTS_ITERATIONS, BATCH_SIZE
from tetris_env import Tetris
from mcts import MCTreeNodeDeterminised
from inference_server import InferenceServer

RANDOM_SEED_MODULUS = 2 ** 32 # Seed methods accept 32-bit integers only

class MCTSAgentEnsemble(MCTSAgent):
    """Ensemble of MCTS agents for Tetris."""

    def __init__(self, checkpoint_name, batch_size=BATCH_SIZE):
        super(MCTSAgentEnsemble, self).__init__(checkpoint_name, batch_size=batch_size)
        # Pre-allocate queues for parallel MCTS simulations. Each worker has its own response queue
        # to avoid contention when receiving responses from the queue.
        self.response_queues = { worker_id: mp.Queue() for worker_id in range(self.n_workers) }
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        # Create and start the synchronous inference server which handles model inference
        # requests sent by the MCTS workers.
        self.inference_server = InferenceServer(
            model=self.model,
            request_queue=self.request_queue,
            response_queues=self.response_queues
        )
        self.inference_server.start()

    def run_episode(self, model, benchmark=False):
        """
        Runs a single episode of an MCTS agent ensemble that explore separate determinised futures
        in parallel and take the majority vote on action selection and tree policy.
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
            chosen_action, available_actions, tree_policy =  run_ensemble_mcts(
                self.env,
                self.request_queue,
                self.response_queues,
                self.result_queue,
                step_count,
                n_workers=self.n_workers,
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

        if benchmark is True:
            return final_score, None, None

        return final_score, transitions, step_count

def run_ensemble_mcts(
    env: Tetris,
    request_queue: mp.Queue,
    response_queues: dict[int, mp.Queue],
    result_queue: mp.Queue,
    step_count: int,
    n_workers: int,
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

    processes = []

    for worker_id in range(n_workers):

        # Each process has its own seed for reproducibility and to prevent the same
        # random number generation (unique in base n_workers)
        process_seed = (step_count * n_workers + worker_id) % RANDOM_SEED_MODULUS

        p = mp.Process(
            target=ensemble_mcts_helper,
            args=(
                env.copy(),
                request_queue,
                response_queues[worker_id],
                worker_id,
                result_queue,
                MCTS_ITERATIONS,
                process_seed
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
        [node["visit_counts"] for node in root_nodes],
        axis=0
    )

    # The next action and tree policy is derived from the aggregate visit counts
    # of actions available to the root nodes in the simulations.

    tree_policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    available_actions = root_nodes[0]["actions"]

    if len(available_actions) == 0:
        # Game ends
        chosen_action = -1
    elif (benchmark is True) or (step_count >= explore_threshold):
        # Greedy when benchmarking or after exploratory steps
        chosen_action = available_actions[np.argmax(mean_visit_counts)]
        tree_policy[chosen_action] = 1.0
    else:
        # Allow exploratory steps at episode start
        mean_visit_counts = mean_visit_counts ** (1 / tau)
        visit_probabilities = mean_visit_counts / np.sum(mean_visit_counts)
        tree_policy[available_actions] = visit_probabilities
        chosen_action = np.random.choice(ACTION_SPACE, p=tree_policy)

    return chosen_action, available_actions, tree_policy

def ensemble_mcts_helper(
    env: Tetris,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_id: int,
    result_queue: mp.Queue,
    iterations: int,
    process_seed: int,
):
    """
    Helper function used by a single process for parallel simulations that runs MCTS
    iterations on a single node and puts the results into the result queue.
    """

    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    random.seed(process_seed)

    root_node = MCTreeNodeDeterminised(
        env=env,
        request_queue=request_queue,
        response_queue=response_queue,
        worker_id=worker_id,
        is_root=True
    )

    for _ in range(iterations):
        root_node.run_iteration()

    result_queue.put({
        "actions": list(root_node.children.keys()),
        "visit_counts": [child.visit_count for child in root_node.children.values()]
    })

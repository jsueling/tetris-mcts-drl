"""
Asynchronous MCTS DRL agent for Tetris using multiple workers for
concurrent shared tree expansion.
"""

import asyncio
import time

from tqdm import tqdm
import numpy as np

from mcts_agent import MCTSAgent, ACTION_SPACE, MCTS_ITERATIONS, BATCH_SIZE
from mcts import MCDecisionNodeAsync
from inference_server import AsyncInferenceServer
from model import A0ResNet

class MCTSAgentAsync(MCTSAgent):
    """Asynchronous MCTS agent for Tetris."""

    def __init__(self, checkpoint_name, batch_size=BATCH_SIZE):
        super(MCTSAgentAsync, self).__init__(checkpoint_name, batch_size=batch_size)
        # Pre-allocate queues for asynchronous tree traversal. Each worker has its
        # own response queue to avoid contention when receiving responses from the queue.
        self.response_queues = { worker_id: asyncio.Queue() for worker_id in range(self.n_workers) }
        self.request_queue = asyncio.Queue()

        # Create the inference server which handles model inference requests sent by workers.
        self.inference_server = AsyncInferenceServer(
            model=self.model,
            request_queue=self.request_queue,
            response_queues=self.response_queues
        )

    def train(self):
        """Wrapper for the asynchronous training loop, overriding the synchronous method."""
        asyncio.run(self._train_async())

    async def _train_async(self):
        """Run the asynchronous training loop for the MCTS agent."""

        # Attempt to restore if there is a saved state
        self.checkpoint.restore_checkpoint()

        # Start inference server within the event loop
        self.inference_server.start()

        # Avoid duplicate benchmarking if there exists a previous benchmark score.
        if len(self.checkpoint.benchmark_scores) == 0:
            for _ in tqdm(range(self.benchmark_episode_count), desc="Benchmarking initial model"):
                final_score, _, _ = await self.run_episode_async(self.model, benchmark=True)
                self.max_benchmark_score += final_score
        else:
            self.max_benchmark_score = self.checkpoint.benchmark_scores[-1]

        starting_iteration = self.checkpoint.completed_iterations

        for iter_idx in tqdm(
            range(starting_iteration, self.total_iterations),
            total=self.total_iterations,
            desc="Training MCTS agent",
            unit="iteration"
        ):

            transitions_added = 0

            for _ in range(self.episodes_per_iteration):

                # Begin episode
                start_time = time.time()

                final_score, transitions, step_count = \
                    await self.run_episode_async(self.model, benchmark=False)

                # End episode
                self.checkpoint.log_episode_results(
                    start_time,
                    final_score,
                    step_count
                )

                self.process_transitions(transitions, final_score)
                transitions_added += len(transitions)

                # Interweave updates with data generation
                if (len(self.buffer) > self.min_buffer_size_for_update) and \
                   (transitions_added >= self.batch_size):
                    self.update()
                    transitions_added -= self.batch_size

            # Use any remaining credit on updates (1:1 ratio of data generation to updates)
            while (len(self.buffer) > self.min_buffer_size_for_update) and \
                  (transitions_added >= self.batch_size):
                self.update()
                transitions_added -= self.batch_size

            # The first iteration is reserved for data generation
            if iter_idx > 0:
                # Evaluates the current model against the candidate model
                # and updates model and benchmark score if the candidate model
                # outperforms the current model.
                await self.evaluate_models_async()

            self.checkpoint.save_iteration(self.model, self.max_benchmark_score)

    async def run_episode_async(self, model, benchmark=False):
        """
        Runs a single episode of an MCTS agent that uses multiple workers
        to expand a shared tree using async/await to mimic threading.
        """

        self.env.reset()

        # Set the current model for the inference server
        self.inference_server.set_model(model)

        # Create the root node of the MCTS tree for this episode which will be iteratively
        # expanded by the workers based on the most promising actions
        root_node = MCDecisionNodeAsync(
            env=self.env.copy(),
            request_queue=self.request_queue,
            response_queues=self.response_queues,
            is_root=True,
        )

        transitions = []
        done = False
        step_count = 0
        while not done:

            await run_async_mcts(
                root_node,
                iterations=MCTS_ITERATIONS,
                n_workers=self.n_workers,
                first_step=(step_count == 0)
            )

            chosen_action, tree_policy = root_node.decide_action(
                tau=0.0 if (benchmark is True) or (step_count >= 30) else 1.0
            )

            if benchmark is False:

                score_before_action = self.env.score
                state_before_action = self.env.get_state()
                legal_actions = np.zeros(ACTION_SPACE, dtype=np.float32)
                legal_actions[list(root_node.chance_node_children.keys())] = 1.0

                transitions.append([
                    state_before_action,
                    tree_policy,
                    legal_actions,
                    score_before_action
                ])

            # Update the environment with the selected action and next Tetromino
            done = self.env.step(chosen_action)

            step_count += 1

            if not done:
                next_tetromino_type = self.env.generate_next_tetromino_type()
                self.env.create_tetromino(next_tetromino_type)

                # Descend the tree to the next node based on the chosen action and randomly
                # generated Tetromino from the real environment, detaching from parent
                root_node = root_node.descend(
                    chosen_action,
                    next_tetromino_type
                )

        final_score = self.env.score

        if benchmark is True:
            return final_score, None, None

        return final_score, transitions, step_count

    async def evaluate_models_async(self):
        """
        Evaluate the current model against the candidate model. The current model
        generates data until a new model outperforms it via benchmarking.
        This happens so that performance is monotonically increasing. This method
        updates the current model to be the best performing model and resets
        the candidate model for the next iteration.
        """

        # Note: Tetris is a single-player game, so only the candidate model needs to be benchmarked
        # against the previously computed max_benchmark_score.

        benchmark_candidate_score = 0
        for _ in range(self.benchmark_episode_count):
            episode_score, _, _ = \
                await self.run_episode_async(model=self.candidate_model, benchmark=True)
            benchmark_candidate_score += episode_score

        # If candidate model outperforms the best model so far, it is adopted for data generation
        if benchmark_candidate_score > self.max_benchmark_score:
            self.max_benchmark_score = benchmark_candidate_score
            self.model.load_state_dict(self.candidate_model.state_dict())

        # In either case, the candidate model is reinstantiated, resetting the
        # optimiser/scheduler but copying the weights
        self.candidate_model = A0ResNet(num_residual_blocks=19, num_actions=ACTION_SPACE)
        self.candidate_model.load_state_dict(self.model.state_dict())

async def run_async_mcts(
    root_node: MCDecisionNodeAsync,
    iterations: int = MCTS_ITERATIONS,
    n_workers=10,
    first_step: bool = False,
):
    """
    Run the MCTS agent with multiple asynchronous workers that update the same
    shared tree. Iterations are divided evenly among the workers.
    """
    iterations_per_worker = iterations // n_workers
    tasks = []
    for worker_id in range(n_workers):

        # At the start of an episode, stagger workers to allow the tree to be expanded slowly,
        # preventing multiple workers from expanding the same node when the tree is still small.
        if first_step is True:
            await asyncio.sleep(0.1)

        tasks.append(
            asyncio.create_task(root_node.run_iterations(worker_id, iterations_per_worker))
        )

    await asyncio.gather(*tasks)

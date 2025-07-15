"""Monte Carlo Tree Search (MCTS) algorithm implementation for Tetris"""

from typing import Dict, Optional
import multiprocessing as mp
from collections import defaultdict
import asyncio

import numpy as np

from tetris_env import Tetris, Tetromino

ACTION_SPACE = 40 # Rotations * Columns
C_PUCT = 1.0  # Hyperparameter modulating prior-guided exploration bonus in pUCT

# Dirichlet noise parameters for exploration, applied to actions at the root node only
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25

class MCTreeNodeDeterminised:
    """
    A node in the tree representing a game state (Tetris grid and current piece).
    Actions are final placements of tetrominoes that connect game states as edges
    (parent-child relation).

    This class implements Determinised Ensemble MCTS, where the majority vote
    of multiple determinised MCTS iterations is used to select the best action
    and as a training target for the neural network.
    """

    def __init__(
        self,
        env: Tetris,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int,
        parent: Optional["MCTreeNodeDeterminised"] = None,
        prior_probability=0.0,
        is_root: bool = False
    ):

        self.env = env
        self.parent: Optional["MCTreeNodeDeterminised"] = parent
        self.is_root = is_root
        self.is_terminal = False
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

        # Backpropagation statistics
        self.visit_count = 0 # N(s, a)
        self.q_value_sum = 0.0 # W(s, a), total action value
        # Q(s, a) is not stored directly, but computed as W(s, a) / N(s, a)

        # The leaf node is expanded and each edge is initialised to
        # { N(s, a) = 0, W(s, a) = 0, Q(s, a) = 0, P(s, a) = p_a }
        # where (v, p_a) = f(s) is the NN evaluation of the previous state,
        # the value is then backed up from leaf to root.

        # == NN priors ==
        # Prior probability will be given by the parent's evaluation during
        # this node's instantiation. It represents the probability of taking
        # the action leading to the state of this node.
        self.prior_probability = prior_probability

        # Each action traverses to child nodes deterministically
        # since the next Tetromino is fixed for all children
        self.children: Dict[int, 'MCTreeNodeDeterminised'] = {}

    def run_iteration(self):
        """
        Run a single iteration of MCTS, consisting of:
        1. Traversing to a leaf node, selecting based on modified UCT
        2. Simulating and expanding the leaf node by NN evaluation
        3. Backing up the results on all nodes from leaf to root
        """
        if self.parent is not None:
            raise RuntimeError("run_iteration must be called on the root node only")
        leaf_node = self.select()
        q_value = leaf_node.evaluate()
        leaf_node.backpropagate(q_value)

    def get_best_child_by_puct(self) -> 'MCTreeNodeDeterminised':
        """
        Returns this node's child with the maximum pUCT value
        (Predictor + Upper Confidence Bound for Trees)
        """
        parent_visit_count = self.visit_count
        children = list(self.children.values())

        visit_counts = np.array([child.visit_count for child in children], dtype=np.float32)
        q_value_sums = np.array([child.q_value_sum for child in children], dtype=np.float32)
        prior_probabilities = np.array(
            [child.prior_probability for child in children],
            dtype=np.float32
        )

        # q_value is an estimate of the expected value of the action,
        # where the child node represents the state immediately following
        # that action. Dividing by visit_count computes a running average
        # of observed q_values. As visit_count increases, q_values should
        # converge to a more accurate estimate of the true value. If child
        # node has not been visited, its q_value is 0 (i.e. not evaluated yet).
        q_value_estimates = np.zeros_like(q_value_sums, dtype=np.float32)
        np.divide(q_value_sums, visit_counts, out=q_value_estimates, where=visit_counts > 0)

        # Balance exploration and exploitation using the modified pUCT formula to recursively
        # select the child node with the highest pUCT value until reaching a leaf node:
        # Q(s, a) + c_puct * P(s, a) * sqrt(∑_b N(s, b)) / (N(s, a) + 1)
        puct_values = q_value_estimates + C_PUCT * prior_probabilities \
            * np.sqrt(parent_visit_count) / (visit_counts + 1)

        return children[np.argmax(puct_values)]

    def select(self) -> 'MCTreeNodeDeterminised':
        """
        Balance exploration and exploitation using the NN-augmented pUCT formula to recursively
        select a child node with the highest pUCT value until reaching a leaf node
        """
        node = self
        # Traverse the tree until reaching a leaf node
        while len(node.children) > 0:
            node = node.get_best_child_by_puct()
        return node

    def nn_evaluation(self):
        """
        This method uses a dual-headed neural network to evaluate the current state.
        It sends the current state to the inference server, blocking until the
        response is received, and returns the policy logits and value.

        Returns:
        - policy_logits: The logits for the action probabilities.
        - value: The estimated value of the current state.
        """
        self.request_queue.put({
            "state": self.env.get_state(),
            "worker_id": self.worker_id
        })
        response = self.response_queue.get()
        policy_logits = response["policy_logits"]
        value = response["value"]
        return policy_logits, value

    def evaluate(self) -> float:
        """
        Use the dual-headed neural network to evaluate the state of this leaf node
        achieving the following:
        1. Simulation/rollout to get a Q-value estimate of the action leading to this state.
        2. Expansion, creating child nodes for all legal actions from this state.
        Returns:
        - q_value: The predicted Q-value of the action leading to this state or -1
        if the state is terminal.
        """

        # If this leaf node is terminal, in Tetris this is always counted as a loss
        # since no further score can be accumulated from a terminal state.
        # Return the minimum q-value possible: tanh from the value head (normalised Tetris scores
        # based on rolling percentile).
        if self.is_terminal:
            return -1

        # Simulation:

        # Generate a mask over the action space of legal actions given the current state
        legal_actions = self.env.get_legal_actions()

        # Leaf nodes with no legal actions cannot be expanded
        if not np.any(legal_actions):
            self.is_terminal = True
            return -1

        # Evaluate as late as possible before expansion
        action_logits, q_value = self.nn_evaluation()

        # Expansion:

        # Mask illegal actions by setting logits to large negative before softmax
        # which will give probability zero after softmax
        action_logits[~legal_actions] = -1e9

        exponential_action_logits = np.exp(action_logits)
        # Apply softmax to action logits to get action probabilities
        action_probabilities = exponential_action_logits / np.sum(exponential_action_logits)

        # Add Dirichlet noise to root node's action probabilities for sufficient exploration
        if self.is_root is True:
            dirichlet_noise = np.random.dirichlet(
                # Noise is distributed among legal actions only
                np.full(np.sum(legal_actions), DIRICHLET_ALPHA)
            )
            action_probabilities[legal_actions] *= (1 - DIRICHLET_EPSILON)
            action_probabilities[legal_actions] += DIRICHLET_EPSILON * dirichlet_noise

        # Determinised sequence of Tetrominoes, all children share the same next Tetromino
        # i.e. all actions share the same stochastic outcome once the node is evaluated.
        next_tetromino_type = self.env.generate_next_tetromino_type()

        for action_index in range(ACTION_SPACE):

            if not legal_actions[action_index]:
                continue

            copy_env = self.env.copy()
            # Place last Tetromino in the copied environment
            _ = copy_env.step(action_index)
            # Create the next Tetromino after the last Tetromino was placed
            copy_env.create_tetromino(next_tetromino_type)

            child_node = MCTreeNodeDeterminised(
                env=copy_env,
                request_queue=self.request_queue,
                response_queue=self.response_queue,
                worker_id=self.worker_id,
                parent=self,
                prior_probability=action_probabilities[action_index]
            )

            # Populate the children dictionary with the action index as key
            # and child node as value
            self.children[action_index] = child_node

        return q_value

    def backpropagate(self, q_value: float):
        """
        Backpropagate the results of the evaluation for all nodes visited in the trajectory,
        traversing parent nodes until reaching the root node.
        """
        node = self
        # Backpropagate the Q-value sums and visit count to all parent nodes
        # after evaluating the leaf node
        while node is not None:
            node.visit_count += 1
            node.q_value_sum += q_value
            node = node.parent

    def decide_action(self, tau=1.0):
        """
        Decide on the best action to take based on the visit counts of immediate child nodes.
        Arguments:
        - tau: Temperature parameter for softmax action selection.
        This parameter modulates exploration vs exploitation in the action selection
        of the actual game.
        Returns:
        - action: The selected action based on the visit counts. \
            Returns -1 if no actions are available.
        - tree_policy: The policy vector for the tree in this state, representing the
        probabilities of selecting each action derived from visit counts of the tree search.
        """

        # Create a tree policy vector representing the probabilities of selecting each action
        # informed by the visit counts of the tree search
        tree_policy = np.zeros(ACTION_SPACE, dtype=np.float32)

        if len(self.children) == 0:
            # If there are no possible actions (no children from the root node),
            # return -1 indicating no action can be taken and a default tree policy
            return -1, tree_policy

        actions = list(self.children.keys())

        if tau == 0.0:
            # If tau is at minimum, select action greedily by the maximum visit count
            chosen_action = max(
                actions,
                key=lambda action: self.children[action].visit_count
            )
            tree_policy[chosen_action] = 1.0
            return chosen_action, tree_policy

        visit_counts = np.array(
            [child.visit_count for child in self.children.values()],
            dtype=np.float32
        )

        # Normalise visit counts using softmax with temperature tau
        visit_counts = visit_counts ** (1 / tau)
        visit_probs = visit_counts / np.sum(visit_counts)
        # Select an action based on the probabilities derived from visit counts
        chosen_action = np.random.choice(actions, p=visit_probs)

        tree_policy[actions] = visit_probs

        return chosen_action, tree_policy

class MCTreeNodeChanceAsync:
    """
    A node in the tree representing a game state (Tetris grid and current piece).
    Actions are final placements of tetrominoes that connect game states as edges
    (parent-child relation).

    This class implements chance nodes for MCTS, where each action splits into
    multiple child nodes representing each possible stochastic outcome of the
    next Tetromino generated by the environment. This class also uses async/await
    to handle the neural network evaluation requests in a non-blocking manner.
    """

    def __init__(
        self,
        env: Tetris,
        request_queue: asyncio.Queue,
        response_queues: dict[int, asyncio.Queue],
        parent: Optional["MCTreeNodeChanceAsync"] = None,
        prior_probability=0.0,
        is_root: bool = False
    ):
        self.env = env
        self.parent = parent
        self.is_root = is_root
        self.is_terminal = False
        self.is_evaluated = False
        self.request_queue = request_queue
        self.response_queues = response_queues

        # Virtual loss for concurrent tree traversal. This discourages multiple workers
        # from exploring the same paths simultaneously, as it temporarily alters the pUCT value
        # of all nodes along the path from root to the leaf node chosen by the worker.
        self.virtual_loss = 0.0

        self.visit_count = 0 # N(s, a)
        self.q_value_sum = 0.0 # W(s, a), total action value
        # Q(s, a) is not stored directly, but computed as W(s, a) / N(s, a)
        self.prior_probability = prior_probability

        # This decision node leads to a chance node for each possible action.
        # Each chance node leads to a decision node for each possible Tetromino type.
        self.successor_nodes: defaultdict[int, list[MCTreeNodeChanceAsync]] = \
            defaultdict(lambda: [None] * (len(Tetromino.figures)))

    async def run_iteration(self, worker_id: int):
        """
        Arguments:
            worker_id (int): The ID of the asynchronous worker used to
            access the correct response queue.
        Run a single asynchronous iteration of MCTS, consisting of:
        1. Traversing to a leaf node, selecting based on modified UCT
        2. Simulating and expanding the leaf node by NN evaluation
        3. Backing up the results on all nodes from leaf to root
        """
        if self.parent is not None:
            raise RuntimeError("run_iteration must be called on the root node only")
        leaf_node = self.select()
        q_value = await leaf_node.evaluate(worker_id)
        leaf_node.backpropagate(q_value)

    async def run_iterations(self, worker_id: int, iterations: int):
        """
        Arguments:
            worker_id (int): The ID of the asynchronous worker used to \
            access the correct response queue.
        Run multiple asynchronous iterations of MCTS.
        """
        for _ in range(iterations):
            await self.run_iteration(worker_id)

    def sample_child_from_best_action_by_puct(self) -> 'MCTreeNodeChanceAsync':
        """
        Returns a child node stochastically sampled from the best pUCT action
        (Predictor + Upper Confidence Bound for Trees).
        """

        parent_visit_count = self.visit_count

        actions, chance_nodes = zip(*self.successor_nodes.items())

        # The sum of visit counts to all child nodes of each chance node
        visit_counts = np.array([
            sum(child.visit_count for child in child_nodes)
            for child_nodes in chance_nodes
        ], dtype=np.float32)

        # The sum of Q-values of all child nodes of each chance node
        q_value_sums = np.array([
            # Subtract virtual loss to discourage exploration of nodes currently being evaluated
            sum((child.q_value_sum - child.virtual_loss) for child in child_nodes)
            for child_nodes in chance_nodes
        ], dtype=np.float32)

        # The child nodes of each chance node share the same prior probability
        prior_probabilities = np.array(
            [child_nodes[0].prior_probability for child_nodes in chance_nodes],
            dtype=np.float32
        )

        # Expected Q-value estimates are a noisy estimate of the value of an action
        # averaged over every stochastic outcome
        expected_q_value_estimates = np.zeros_like(q_value_sums, dtype=np.float32)
        np.divide(
            q_value_sums,
            visit_counts,
            out=expected_q_value_estimates,
            where=visit_counts > 0
        )

        # Q(s, a) + c_puct * P(s, a) * sqrt(∑_b N(s, b)) / (N(s, a) + 1)
        puct_values = expected_q_value_estimates + C_PUCT * prior_probabilities \
            * np.sqrt(parent_visit_count) / (visit_counts + 1)

        best_puct_action = actions[np.argmax(puct_values)]
        # Randomly select one of the possible Tetrominoes
        random_tetromino_index = np.random.randint(0, len(Tetromino.figures))
        # Stochastically transition to a child node based on the best pUCT action
        return self.successor_nodes[best_puct_action][random_tetromino_index]

    def select(self) -> 'MCTreeNodeChanceAsync':
        """
        Balance exploration and exploitation using the NN-augmented pUCT formula to recursively
        sample a child node from the highest pUCT-valued action until reaching a leaf node
        """
        node = self
        while True:

            # Add virtual loss to all nodes along the current path
            # to discourage exploration of nodes currently being evaluated
            # by other workers.
            node.virtual_loss += 1

            # Continue selection until a leaf node is found,
            # a leaf node is either not evaluated or terminal
            if not node.is_evaluated or node.is_terminal:
                return node

            # Transition stochastically to a child node based on the best pUCT action.
            node = node.sample_child_from_best_action_by_puct()

    async def nn_evaluation(self, worker_id: int):
        """
        This method uses a dual-headed neural network to evaluate the current state.
        It sends the current state to the inference server, awaiting until the
        response is received, and returns the policy logits and value.

        Returns:
        - policy_logits: The logits for the action probabilities.
        - value: The estimated value of the current state.
        """
        await self.request_queue.put({
            "state": self.env.get_state(),
            "worker_id": worker_id
        })
        response = await self.response_queues[worker_id].get()
        policy_logits = response["policy_logits"]
        value = response["value"]
        return policy_logits, value

    async def evaluate(self, worker_id: int) -> float:
        """
        Use the dual-headed neural network to evaluate the state of this leaf node
        achieving the following:
        1. Simulation/rollout to get a Q-value estimate of the action leading to this state.
        2. Expansion, creating successor nodes for all legal actions from this state.
        Returns:
        - q_value: The predicted Q-value of the action leading to this state or -1
        if the state is terminal.
        """

        if self.is_terminal:
            return -1

        # Simulation:

        legal_actions = self.env.get_legal_actions()

        if not np.any(legal_actions):
            self.is_terminal = True
            return -1

        action_logits, q_value = await self.nn_evaluation(worker_id)

        # Expansion:

        action_logits[~legal_actions] = -1e9

        exponential_action_logits = np.exp(action_logits)
        action_probabilities = exponential_action_logits / np.sum(exponential_action_logits)

        if self.is_root is True:
            dirichlet_noise = np.random.dirichlet(
                np.full(np.sum(legal_actions), DIRICHLET_ALPHA)
            )
            action_probabilities[legal_actions] *= (1 - DIRICHLET_EPSILON)
            action_probabilities[legal_actions] += DIRICHLET_EPSILON * dirichlet_noise

        for action_index in range(ACTION_SPACE):

            if not legal_actions[action_index]:
                continue

            for next_tetromino_type in range(len(Tetromino.figures)):

                copy_env = self.env.copy()
                # Place last Tetromino in the copied environment
                _ = copy_env.step(action_index)
                # Create the next Tetromino after the last Tetromino was placed
                copy_env.create_tetromino(next_tetromino_type)

                child_node = MCTreeNodeChanceAsync(
                    env=copy_env,
                    request_queue=self.request_queue,
                    response_queues=self.response_queues,
                    parent=self,
                    prior_probability=action_probabilities[action_index]
                )

                # Populate the successor nodes dictionary
                self.successor_nodes[action_index][next_tetromino_type] = child_node

        # Setting evaluated to True any earlier might lead to other workers
        # incorrectly traversing to non-existent children of this node, throwing an error.
        # In the worst case, having multiple evaluations of the same node is
        # the lesser problem, as subsequent evaluations will overwrite the children
        # of the first evaluation but could inflate q_value sums/visit counts
        # by a small amount via backpropagation

        self.is_evaluated = True
        return q_value

    def backpropagate(self, q_value: float):
        """
        Backpropagate the results of the evaluation for all nodes visited in the trajectory,
        traversing parent nodes until reaching the root node.
        """
        node = self
        while node is not None:

            # Remove virtual loss during backpropagation
            node.virtual_loss -= 1

            node.visit_count += 1
            node.q_value_sum += q_value
            node = node.parent

    def decide_action(self, tau=1.0):
        """
        Decide on the best action to take based on the visit counts of immediate child nodes.
        Arguments:
        - tau: Temperature parameter for softmax action selection.
        This parameter modulates exploration vs exploitation in the action selection
        of the actual game.
        Returns:
        - action: The selected action based on the visit counts. \
            Returns -1 if no actions are available.
        - tree_policy: The policy vector for the tree in this state, representing the
        probabilities of selecting each action derived from visit counts of the tree search.
        """

        tree_policy = np.zeros(ACTION_SPACE, dtype=np.float32)

        if len(self.successor_nodes) == 0:
            return -1, tree_policy

        actions, chance_nodes = zip(*self.successor_nodes.items())

        # The sum of visit counts to all child nodes of each chance node
        visit_counts = np.array([
            sum(child.visit_count for child in child_nodes)
            for child_nodes in chance_nodes
        ], dtype=np.float32)

        if tau == 0.0:
            chosen_action = actions[np.argmax(visit_counts)]
            tree_policy[chosen_action] = 1.0
            return chosen_action, tree_policy

        # Normalise visit counts using softmax with temperature tau
        visit_counts = visit_counts ** (1 / tau)
        visit_probs = visit_counts / np.sum(visit_counts)
        # Select an action based on the probabilities derived from visit counts
        chosen_action = np.random.choice(actions, p=visit_probs)

        tree_policy[np.array(actions)] = visit_probs

        return chosen_action, tree_policy

    def descend(
        self,
        action: int,
        tetromino_type: int
    ) -> 'MCTreeNodeChanceAsync':
        """
        Descend to the child node corresponding to the given action and Tetromino type
        resulting from interaction with the real environment. Detach the new root from
        its parent node. This allows reuse of the tree statistics computed from previous
        iterations

        Returns:
        - successor: The new root node of the subtree corresponding
        to the action and Tetromino type.
        """
        successor = self.successor_nodes[action][tetromino_type]
        successor.parent = None
        return successor

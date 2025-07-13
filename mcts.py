"""Monte Carlo Tree Search (MCTS) algorithm implementation for Tetris"""

from typing import Dict, Optional
import multiprocessing as mp

import numpy as np

from tetris_env import Tetris

ACTION_SPACE = 40 # Rotations * Columns
C_PUCT = 1.0  # Hyperparameter modulating prior-guided exploration bonus in pUCT

# Dirichlet noise parameters for exploration, applied to actions at the root node only
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25

class MonteCarloTreeNode:
    """
    A node in the tree representing a game state (Tetris grid and current piece).
    Actions are final placements of tetrominoes that connect game states as edges
    (parent-child relation).
    """

    def __init__(
        self,
        env: Tetris,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_id: int,
        parent: Optional["MonteCarloTreeNode"] = None,
        prior_probability=0.0,
        is_root: bool = False
    ):

        self.env = env
        self.parent: Optional["MonteCarloTreeNode"] = parent
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

        self.children: Dict[int, 'MonteCarloTreeNode'] = {}

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

    def get_best_child_by_puct(self) -> 'MonteCarloTreeNode':
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

    def select(self) -> 'MonteCarloTreeNode':
        """
        Traverse down the tree based on maximum pUCT value until
        reaching a leaf node.
        """
        node = self
        # Traverse the tree until reaching a leaf node
        while len(node.children) > 0:
            node = node.get_best_child_by_puct()
        return node

    def nn_evaluation(self):
        """
        This method uses a dual-headed neural network to evaluate the current state.
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
        - q_value: The predicted Q-value of the action leading to this state or the score
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
            _, _ = copy_env.step(action_index)
            # Create the next Tetromino after the last Tetromino was placed
            copy_env.create_tetromino(next_tetromino_type)

            child_node = MonteCarloTreeNode(
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

"""Monte Carlo Tree Search (MCTS) algorithm implementation."""

import numpy as np

from tetris_env import Tetris

ACTION_SPACE = 40 # Rotations * Columns
C_PUCT = 1.0  # Hyperparameter modulating prior-guided exploration bonus in pUCT

class MonteCarloTreeNode:
    """
    A node in the tree representing a game state.
    Actions connect game states as edges (parent-child relation).
    """
    def __init__(self, state: Tetris, parent=None, prior_probability=0.0):

        self.state = state
        self.parent: MonteCarloTreeNode = parent

        # Backpropagation statistics
        self.visit_count = 0
        self.q_value_sum = 0.0

        # NN priors
        # Prior probability will be given by the parent's evaluation
        # during this node's instantiation
        self.prior_probability = prior_probability
        # Prior Q-value will be zero until this node is evaluated.
        self.prior_q_value = 0.0
        self.is_evaluated = False
        self.children: list[MonteCarloTreeNode] = []

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

    def _puct_value(self, child: 'MonteCarloTreeNode') -> float:
        """
        Calculate the pUCT (Predictor + Upper Confidence Bound for Trees) value
        for a given node.
        """

        parent_visit_count = self.visit_count

        # q_value is an estimate of the expected value of the action,
        # where the child node represents the state immediately following
        # that action. Dividing by visit_count computes a running average
        # of observed q_values. As visit_count increases, q_values should
        # converge to a more accurate estimate of the true value. If child
        # node has not been visited, its q_value is 0 (i.e. not evaluated yet).
        q_value_estimate = (child.q_value_sum / child.visit_count) if child.visit_count > 0 else 0.0

        # Balance exploration and exploitation using the modified pUCT formula:
        # Q(s, a) + c_puct * P(s, a) * sqrt(∑_b N(s, b)) / (N(s, a) + 1)
        puct_value = q_value_estimate + C_PUCT * child.prior_probability \
            * np.sqrt(parent_visit_count) / (child.visit_count + 1)

        return puct_value

    def select(self) -> 'MonteCarloTreeNode':
        """
        Traverse down the tree based on maximum pUCT value until
        reaching a leaf node.
        """
        node = self
        # Traverse the tree until reaching a leaf node
        while len(node.children) > 0:
            # Select the child with the highest pUCT value
            node = max(node.children, key=self._puct_value)
        return node

    def nn_evaluation(self):
        """
        Use Neural Network to evaluate the current state
        """
        # TODO Placeholder neural network call
        # returns a tuple of (Q-value, action_probabilities)
        return self.model(self.state.grid)

    def evaluate(self) -> float:
        """
        Use the dual-headed neural network to evaluate the current state achieving the following:
        1. Simulation/rollout to get a Q-value estimate of the action leading to this state.
        2. Expansion, creating child nodes for all legal actions from this state.
        """

        # Tree policy has reached a leaf node which is terminal and has been evaluated.
        # Simulation is skipped
        # Return the actual score as the terminal value
        if self.is_evaluated is True:
            return self.state.score

        # Simulation:

        # Generate all legal actions from the current state
        legal_actions = self.state.get_legal_actions()
        q_value, action_logits = self.nn_evaluation()
        self.prior_q_value = q_value

        # Tree policy has reached a leaf node which is terminal but has not yet
        # been evaluated. Set evaluated to true and return the actual score as the terminal value
        if not legal_actions:
            self.is_evaluated = True
            return self.state.score

        # Expansion:

        # Mask illegal actions by setting their probabilities to -inf before softmax
        for action_index in range(ACTION_SPACE):
            illegal_action = not legal_actions[action_index]
            if illegal_action:
                action_logits[action_index] = float("-inf")

        exponential_action_logits = np.exp(action_logits)
        # Apply softmax to action logits to get action probabilities
        action_probabilities = exponential_action_logits / np.sum(exponential_action_logits)

        for action_index in range(ACTION_SPACE):

            if not legal_actions[action_index]:
                continue

            # TODO Handle creating next state
            next_state = self.state.step(action_index)

            # TODO Handle next Tetromino piece selection

            prior_probability = action_probabilities[action_index]

            child_node = MonteCarloTreeNode(
                state=next_state,
                parent=self,
                prior_probability=prior_probability
            )

            self.children.append(child_node)

        # Mark this node as evaluated
        self.is_evaluated = True
        return q_value

    def backpropagate(self, q_value: float):
        """
        Backpropagate the results of the evaluation for all nodes visited in the trajectory,
        traversing parent nodes until reaching the root node.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.q_value_sum += q_value
            node = node.parent

    def detach(self):
        """Detach this node from its parent, effectively removing it from the tree."""
        del self.parent
        self.parent = None

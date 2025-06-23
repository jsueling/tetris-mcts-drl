"""Monte Carlo Tree Search (MCTS) algorithm implementation."""

import numpy as np

ACTION_SPACE = 40 # Rotations * Columns
C_PUCT = 1.0  # Hyperparameter modulating prior-guided exploration bonus in pUCT

class MonteCarloTreeNode:
    """
    A node in the tree representing a game state.
    Actions connect game states as edges (parent-child relation).
    """
    def __init__(self, state, parent=None, prior_probability=0.0, prior_value=0.0):

        self.state = state
        self.parent: MonteCarloTreeNode = parent

        # Backup statistics
        self.visit_count = 0
        self.value_sum = 0.0

        # NN priors
        # Prior probability will be given by the parent's expansion
        # during this node's instantiation
        self.prior_probability = prior_probability
        # Prior value will be zero until this node is expanded.
        self.prior_value = prior_value

        # Unexpanded nodes are equivalent to leaf nodes in the tree.
        self.is_expanded = False
        self.children: list[MonteCarloTreeNode] = []

    def run_iteration(self):
        """
        Run a single iteration of MCTS, consisting of:
        Selecting a leaf node, expanding it and backing up the results.
        """
        leaf_node = self.select()
        leaf_node.expand()
        leaf_node.backup()

    def _puct_value(self, child: 'MonteCarloTreeNode' | None) -> float:
        """
        Calculate the pUCT (Predictor + Upper Confidence Bound for Trees) value
        for a given node.
        """

        if child is None:
            return float('-inf')

        parent_visit_count = self.visit_count

        # Balance exploration and exploitation using the modified pUCT formula:
        # Q(s, a) + c_puct * P(s, a) * sqrt(∑_b N(s, b)) / (N(s, a) + 1)
        puct_value = child.prior_value + C_PUCT * child.prior_probability \
            * np.sqrt(parent_visit_count) / (child.visit_count + 1)

        return puct_value

    def select(self) -> 'MonteCarloTreeNode':
        """
        Traverse down the tree based on maximum pUCT value until
        reaching a leaf node.
        """
        node = self
        # Traverse the tree until reaching a leaf node
        while node.is_expanded:
            # Select the child with the highest pUCT value
            node = max(node.children, key=self._puct_value)
        return node

    def rollout(self):
        """
        Use Neural Network to evaluate the current state
        """
        pass

    def expand(self):
        """Expand the current node by creating child nodes for all legal actions."""
        pass

    def backup(self):
        """
        Backup the result for all nodes visited in the trajectory, traversing parent nodes
        until reaching the root node.
        """
        pass
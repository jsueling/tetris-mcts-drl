import numpy as np

ACTION_SPACE_SIZE = 40 # Rotations * Columns

class MonteCarloTreeNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree representing a game state.
    """
    def __init__(self, state, parent=None, prior_probability=0.0, prior_value=0.0):

        self.state = state
        self.parent: MonteCarloTreeNode = parent
        self.visit_count = 0
        self.value_sum = 0.0
        # NN priors
        self.prior_probability = prior_probability
        self.value_sum += prior_value

        # Expanded nodes are equivalent to leaf nodes in the tree.
        self.is_expanded = False
        self.children: dict[int, MonteCarloTreeNode] = {}
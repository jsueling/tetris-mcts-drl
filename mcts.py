"""Monte Carlo Tree Search (MCTS) algorithm implementation."""

from typing import Dict, Optional

import torch
import numpy as np

from tetris_env import Tetris

ACTION_SPACE = 40 # Rotations * Columns
C_PUCT = 1.0  # Hyperparameter modulating prior-guided exploration bonus in pUCT

# Dirichlet noise parameters for exploration
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25

class MonteCarloTreeNode:
    """
    A node in the tree representing a game state.
    Actions connect game states as edges (parent-child relation).
    """

    def __init__(
        self,
        env: Tetris,
        parent: Optional["MonteCarloTreeNode"] = None,
        is_root: bool = False,
        prior_probability=0.0,
        model=None
    ):

        self.env = env
        self.parent: Optional["MonteCarloTreeNode"] = parent
        self.model = model
        self.is_root = is_root

        # Backpropagation statistics
        self.visit_count = 0
        self.q_value_sum = 0.0

        # NN priors
        # Prior probability will be given by the parent's evaluation
        # during this node's instantiation
        self.prior_probability = prior_probability
        # Prior Q-value will be zero until this node is evaluated.
        self.q_value = 0.0
        self.is_evaluated = False

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
        (Predictor + Upper Confidence Bound for Trees),
        normalising q-value estimates into the range [0, 1].
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

        # Normalise q_value_estimates to [0, 1] range since the network outputs
        # raw game scores (lines cleared) which must be normalised for balance
        # between exploration and exploitation terms.
        max_q_value_estimate = np.max(q_value_estimates)
        min_q_value_estimate = np.min(q_value_estimates)
        q_value_estimates_normalised = (q_value_estimates - min_q_value_estimate) \
            / (max_q_value_estimate - min_q_value_estimate + 1e-8)

        # Balance exploration and exploitation using the modified pUCT formula:
        # Q(s, a) + c_puct * P(s, a) * sqrt(∑_b N(s, b)) / (N(s, a) + 1)
        puct_values = q_value_estimates_normalised + C_PUCT * prior_probabilities \
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

        # Single inference is very inefficient.
        # TODO: Batch multiple inferences together

        self.model.eval()
        with torch.no_grad():
            state = self.env.get_state()
            state_gpu = torch.tensor(
                np.expand_dims(state, axis=0), # Add batch dimension
                dtype=torch.float32
            ).to(self.model.device)
            # Forward pass through the neural network
            policy_logits, value = self.model(state_gpu)
        return policy_logits, value

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
            return self.env.score

        # Simulation:

        # Generate all legal actions from the current state
        legal_actions = self.env.get_legal_actions()

        # Tree policy has reached a leaf node which is terminal but has not yet
        # been evaluated. Set evaluated to true and return the actual score as the terminal value
        if not np.any(legal_actions):
            self.is_evaluated = True
            return self.env.score

        # Evaluate as late as possible before expansion
        action_logits, q_value = self.nn_evaluation()

        action_logits = action_logits.squeeze(0).cpu().numpy()
        q_value = q_value.item()

        # Expansion:

        # Mask illegal actions by setting logits to large negative before softmax
        # which will give probability zero after softmax
        action_logits[~legal_actions] = -1e9

        exponential_action_logits = np.exp(action_logits)
        # Apply softmax to action logits to get action probabilities
        action_probabilities = exponential_action_logits / np.sum(exponential_action_logits)

        # Add Dirichlet noise to root node's action probabilities for sufficient exploration
        if self.is_root:
            dirichlet_noise = np.random.dirichlet(
                # Noise is distributed among legal actions only
                np.full(np.sum(legal_actions), DIRICHLET_ALPHA)
            )
            action_probabilities[legal_actions] *= (1 - DIRICHLET_EPSILON)
            action_probabilities[legal_actions] += DIRICHLET_EPSILON * dirichlet_noise

        # Determinise the generation of the next Tetromino for all child nodes.
        # This means all actions share the same stochastic outcome once the node is evaluated.
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
                parent=self,
                prior_probability=action_probabilities[action_index],
                model=self.model
            )

            self.children[action_index] = child_node

        # Mark this node as evaluated
        self.is_evaluated = True
        return q_value

    def backpropagate(self, q_value: float):
        """
        Backpropagate the results of the evaluation for all nodes visited in the trajectory,
        traversing parent nodes until reaching the root node.
        """
        node = self
        # Set initial Q-value for the node being evaluated
        node.q_value = q_value
        # Backpropagate the Q-value sums and visit count to all parent nodes
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
            # return -1 indicating no action can be taken and an empty tree policy
            return -1, tree_policy

        if tau == 0.0:
            # If tau is at minimum, select action greedily by the maximum visit count
            action = max(self.children, key=lambda action: self.children[action].visit_count)
            tree_policy[action] = 1.0
            return action, tree_policy

        actions = list(self.children.keys())
        visit_counts = np.array(
            [child.visit_count for child in self.children.values()],
            dtype=np.float32
        )

        # Normalise visit counts using softmax with temperature tau
        visit_counts = visit_counts ** (1 / tau)
        visit_probs = visit_counts / np.sum(visit_counts)
        # Select an action based on the probabilities derived from visit counts
        action = np.random.choice(actions, p=visit_probs)

        tree_policy[actions] = visit_probs

        return action, tree_policy

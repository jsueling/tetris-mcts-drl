"""
Monte Carlo Tree Search (MCTS) algorithm implementation for Tetris. There are two implementations:
1. Determinised Ensemble MCTS (MCTreeNodeDeterminised) - \
    uses separate trees per worker each with a separate determinised sequence of Tetrominoes.
2. MCTS with Decision/Chance nodes - similar to single player Expectimax Search \
    (MCDecisionNodeAsync + ChanceNode) - uses async/await for concurrent shared tree expansion.
"""

from typing import Optional
import multiprocessing as mp
import asyncio

import numpy as np
import torch

from tetris_env import Tetris, Tetromino

ACTION_SPACE = 40 # Rotations * Columns
C_PUCT = 1.0  # Hyperparameter modulating prior-guided exploration bonus in PUCT

# Dirichlet noise parameters for exploration, applied to actions at the root node only
DIRICHLET_EPSILON = 0.25
# Dirichlet alpha is chosen proportional to the average number of legal moves,
# 0.03 was used in AlphaZero for Chess
DIRICHLET_ALPHA = 0.03

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
        shared_buffer: Optional[torch.Tensor] = None,
        parent: Optional["MCTreeNodeDeterminised"] = None,
        is_root: bool = False,
        prev_action_idx: Optional[int] = None
    ):

        self.env = env
        self.parent: Optional["MCTreeNodeDeterminised"] = parent
        self.is_root = is_root
        self.is_terminal = False
        self.is_evaluated = False

        # Communication with inference server
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self.shared_buffer = shared_buffer

        # Backpropagation statistics used for selection of this node's children,
        # lazily allocated at expansion
        self.available_actions = None
        self.children = None
        self.prior_probabilities = None
        self.q_value_sums = None
        self.total_visit_count = 0
        self.visit_counts = None
        # Determinised sequence - all children will share the same Tetromino type
        self.next_tetromino_type = self.env.generate_next_tetromino_type()
        # Reference to the action taken to reach this node, used in backpropagation
        self.prev_action_index = prev_action_idx

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

    def get_best_action_by_puct(self) -> 'int':
        """
        Returns this node's maximum PUCT value action
        (Predictor + Upper Confidence Bound for Trees)
        """

        # ∑_b N(s, b)
        parent_visit_count = self.total_visit_count
        # N(s, a) for each child node
        visit_counts = self.visit_counts[self.available_actions]
        # W(s, a) for each child node
        q_value_sums = self.q_value_sums[self.available_actions]
        # P(s, a) for each child node
        prior_probabilities = self.prior_probabilities[self.available_actions]

        # q_value is an estimate of the expected value of the action,
        # where the child node represents the state immediately following
        # that action. Dividing by visit_count computes a running average
        # of observed q_values. As visit_count increases, q_values should
        # converge to a more accurate estimate of the true value. If child
        # node has not been visited, its q_value is 0 (i.e. not evaluated yet).
        q_value_estimates = np.zeros_like(q_value_sums, dtype=np.float32)

        # Q(s, a) = W(s, a) / N(s, a)
        np.divide(
            q_value_sums,
            visit_counts,
            out=q_value_estimates,
            where=visit_counts > 0
        )

        # Balance exploration and exploitation using the modified PUCT formula to recursively
        # select the child node with the highest PUCT value until reaching a leaf node:
        # Q(s, a) + c_puct * P(s, a) * sqrt(∑_b N(s, b)) / (N(s, a) + 1)
        puct_values = q_value_estimates + C_PUCT * prior_probabilities \
            * np.sqrt(parent_visit_count) / (visit_counts + 1)

        # Randomly select action among max PUCT value actions

        max_puct_indices = []
        max_puct_value = float('-inf')
        for i, puct_value in enumerate(puct_values):
            if puct_value > max_puct_value:
                max_puct_value = puct_value
                max_puct_indices = [i]
            elif puct_value == max_puct_value:
                max_puct_indices.append(i)

        return self.available_actions[np.random.choice(max_puct_indices)]

    def select(self) -> 'MCTreeNodeDeterminised':
        """
        Balance exploration and exploitation using the NN-augmented PUCT formula to recursively
        select a child node with the highest PUCT value until reaching a leaf node

        This method JIT-initialises nodes to save memory and time by only creating nodes
        when they are needed.
        """
        node = self
        # Traverse the tree until reaching a leaf node
        while node.is_evaluated and not node.is_terminal:
            chosen_action_idx = node.get_best_action_by_puct()
            # JIT-initialise the child node if it does not exist yet
            if node.children[chosen_action_idx] is None:
                copy_env = node.env.copy()
                copy_env.step(chosen_action_idx)
                # Determinised sequence of Tetrominoes, all children share the same next Tetromino
                copy_env.create_tetromino(node.next_tetromino_type)
                node.children[chosen_action_idx] = MCTreeNodeDeterminised(
                    env=copy_env,
                    request_queue=node.request_queue,
                    response_queue=node.response_queue,
                    worker_id=node.worker_id,
                    shared_buffer=node.shared_buffer,
                    parent=node,
                    prev_action_idx=chosen_action_idx,
                )
            node = node.children[chosen_action_idx]
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
        state_tensor = torch.tensor(
            self.env.get_state(),
            dtype=torch.float32
        )
        # Copy data to shared memory
        self.shared_buffer.copy_(state_tensor)
        # Passing shared memory tensors through the queue is a no-op:
        # https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#reuse-buffers-passed-through-a-queue
        self.request_queue.put({
            "state": self.shared_buffer,
            "worker_id": self.worker_id
        })
        response = self.response_queue.get()
        policy_logits = response["policy_logits"]
        value = response["value"]
        self.shared_buffer = response["shared_buffer"]
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

        # Numerically stable softmax computation to avoid overflow with large logits
        exponential_action_logits = np.exp(action_logits - np.max(action_logits))
        action_probabilities = exponential_action_logits / np.sum(exponential_action_logits)

        # Add Dirichlet noise to root node's action probabilities for sufficient exploration
        if self.is_root is True:
            dirichlet_noise = np.random.dirichlet(
                # Noise is distributed among legal actions only
                np.full(np.sum(legal_actions), DIRICHLET_ALPHA)
            )
            action_probabilities[legal_actions] *= (1 - DIRICHLET_EPSILON)
            action_probabilities[legal_actions] += DIRICHLET_EPSILON * dirichlet_noise

        # Creation of leaf node objects is deferred until selection where the
        # objects are JIT-initialised

        # Each leaf node is initialised to:
        # { N(s, a) = 0, W(s, a) = 0, Q(s, a) = 0, P(s, a) = p_a }
        # where (v, p_a) = f(s) is the NN evaluation of the previous node (s),

        self.available_actions = np.where(legal_actions)[0]
        self.children = np.empty(ACTION_SPACE, dtype=object)
        self.visit_counts = np.zeros(ACTION_SPACE, dtype=np.float32)
        self.q_value_sums = np.zeros(ACTION_SPACE, dtype=np.float32)
        self.prior_probabilities = action_probabilities

        self.is_evaluated = True
        return q_value

    def backpropagate(self, q_value: float):
        """
        Backpropagate the results of the evaluation for all nodes visited in the trajectory,
        traversing parent nodes until reaching the root node.
        """
        # Increment leaf node's total visit count
        self.total_visit_count += 1
        node = self
        while node.parent:
            # The action leading to this node
            prev_action_index = node.prev_action_index
            parent = node.parent
            # Statistics are stored in the parent node
            parent.total_visit_count += 1 # ∑_b N(s, b)
            parent.visit_counts[prev_action_index] += 1 # N(s, a)
            parent.q_value_sums[prev_action_index] += q_value # W(s, a)
            node = parent

class ChanceNode:
    """
    A chance node connecting decision nodes in the MCTS tree. The parent of
    this node is a decision node where the action taken leads to this chance
    node. Each child node is one of the possible stochastic outcomes of that
    action (next Tetromino generated by the environment).
    """
    def __init__(self, parent: "MCDecisionNodeAsync"):

        self.parent = parent

        # Each chance node leads to a decision node for each possible Tetromino type.
        self.decision_node_children: np.ndarray[MCDecisionNodeAsync] = \
            np.empty(len(Tetromino.figures), dtype=object)

        self.visit_counts = np.zeros(len(Tetromino.figures), dtype=np.float32)

class MCDecisionNodeAsync:
    """
    A decision node in the tree representing a game state (Tetris grid and current piece).
    Actions are final placements of tetrominoes that connect decision nodes to chance nodes,
    which then lead to other decision nodes representing the outcome of the action
    (parent-child relations). The children of chance nodes represent all possible
    stochastic outcomes of each action available to the top-level decision node.

    This class uses async/await to handle the neural network evaluation requests
    in a non-blocking manner thus allowing concurrent tree expansion.
    """

    def __init__(
        self,
        env: Tetris,
        request_queue: asyncio.Queue,
        response_queues: dict[int, asyncio.Queue],
        parent: Optional["ChanceNode"] = None,
        is_root: bool = False,
        prev_action_idx: Optional[int] = None
    ):
        self.env = env
        self.parent = parent
        self.is_root = is_root
        self.is_terminal = False
        self.is_evaluated = False
        self.request_queue = request_queue
        self.response_queues = response_queues

        # The action index taken to reach this node from the grandparent decision node.
        self.prev_action_index = prev_action_idx

        # Sum of the visit counts of all actions taken from this node.
        self.total_visit_count = 0 # ∑_b N(s, b)

        # This decision node leads to a chance node for each possible action.
        self.chance_node_children: None

        # Stores available actions for fast slicing. Computed at expansion
        # for use in PUCT selection thereafter.
        self.visit_counts = None
        self.q_value_sums = None
        self.prior_probabilities = None
        self.available_actions: np.ndarray = None

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

    def get_best_action_by_puct(self) -> 'int':
        """
        Returns the index of the action with the best expected PUCT score based on
        that node's children (Predictor + Upper Confidence Bound for Trees).
        """

        # ∑_b N(s, b)
        parent_visit_count = self.total_visit_count
        # W(s, a) for each child node
        q_value_sums = self.q_value_sums[self.available_actions]
        # N(s, a) for each child node
        visit_counts = self.visit_counts[self.available_actions]
        # P(s, a) for each child node
        prior_probabilities = self.prior_probabilities[self.available_actions]

        # Expected Q-value estimates are a noisy estimate of the value of an action
        # averaged over every stochastic outcome

        # Q(s, a) = W(s, a) / N(s, a)
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

        # Randomly select action among max PUCT value actions

        max_puct_value = float("-inf")
        max_puct_action_indices = []
        for i, puct_value in enumerate(puct_values):
            if puct_value > max_puct_value:
                max_puct_value = puct_value
                max_puct_action_indices = [i]
            elif puct_value == max_puct_value:
                max_puct_action_indices.append(i)

        return self.available_actions[np.random.choice(max_puct_action_indices)]

    def select(self) -> 'MCDecisionNodeAsync':
        """
        Balance exploration and exploitation using the NN-augmented PUCT formula to recursively
        sample a child node from the highest PUCT-valued action until reaching a leaf node.

        This method JIT-initialises nodes to save memory and time by only creating nodes
        as needed during selection.

        Returns:
        - decision_node: The leaf decision node reached by the selection process which is the first
        node that is either not evaluated or is a terminal node.
        """

        decision_node = self

        # Traverse to a leaf node
        while decision_node.is_evaluated and not decision_node.is_terminal:

            # Decide on action, transitioning to chance node, based on the best PUCT value
            chosen_action_idx = decision_node.get_best_action_by_puct()

            # Add virtual loss to all actions taken along the current path to to discourage
            # duplicate evaluation of nodes currently being evaluated by other workers,
            # affecting PUCT during selection

            # Virtual visit discourage exploration term of PUCT
            decision_node.visit_counts[chosen_action_idx] += 1
            # Virtual loss of -1.0 discourage exploitation term of PUCT
            decision_node.q_value_sums[chosen_action_idx] -= 1.0

            decision_node.total_visit_count += 1

            # Chance node object, JIT-initialised when needed
            if decision_node.chance_node_children[chosen_action_idx] is None:
                decision_node.chance_node_children[chosen_action_idx] = \
                    ChanceNode(parent=decision_node)

            # Traverse to chance node
            chance_node = decision_node.chance_node_children[chosen_action_idx]

            # Stochastic outcome, transitioning to a decision node. Here we choose
            # a child with minimum visit count selected randomly to converge to expected
            # values faster since the distribution of tetrominoes is randomly uniform.

            min_visits = float('inf')
            min_visit_indices = []
            for i, visit_count in enumerate(chance_node.visit_counts):
                if visit_count < min_visits:
                    min_visits = visit_count
                    min_visit_indices = [i]
                elif visit_count == min_visits:
                    min_visit_indices.append(i)

            random_min_visited_tetromino = np.random.choice(min_visit_indices)

            # Decision node object, JIT-initialised when needed
            if chance_node.decision_node_children[random_min_visited_tetromino] is None:

                copy_env = decision_node.env.copy()
                copy_env.step(chosen_action_idx)
                copy_env.create_tetromino(random_min_visited_tetromino)

                chance_node.decision_node_children[random_min_visited_tetromino] = \
                MCDecisionNodeAsync(
                    env=copy_env,
                    request_queue=self.request_queue,
                    response_queues=self.response_queues,
                    parent=chance_node,
                    prev_action_idx=chosen_action_idx
                )

            chance_node.visit_counts[random_min_visited_tetromino] += 1

            # Traverse to decision node
            decision_node = chance_node.decision_node_children[random_min_visited_tetromino]

        # Set the total visit count of the leaf node to 1
        # The virtual loss visit is added here and then not decremented in backpropagation
        decision_node.total_visit_count += 1
        return decision_node

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

        available_actions = np.where(legal_actions)[0]

        if available_actions.size == 0:
            self.is_terminal = True
            return -1

        action_logits, q_value = await self.nn_evaluation(worker_id)

        # Expansion:

        action_logits[~legal_actions] = -1e9

        # Numerically stable softmax computation to avoid overflow with large logits
        exponential_action_logits = np.exp(action_logits - np.max(action_logits))
        action_probabilities = exponential_action_logits / np.sum(exponential_action_logits)

        if self.is_root is True:
            dirichlet_noise = np.random.dirichlet(
                np.full(np.sum(legal_actions), DIRICHLET_ALPHA)
            )
            action_probabilities[legal_actions] *= (1 - DIRICHLET_EPSILON)
            action_probabilities[legal_actions] += DIRICHLET_EPSILON * dirichlet_noise

        # Creation of child node objects is deferred until selection where nodes
        # are JIT-initialised

        self.visit_counts = np.zeros(ACTION_SPACE, dtype=np.float32)
        self.q_value_sums = np.zeros(ACTION_SPACE, dtype=np.float32)
        self.available_actions = available_actions
        self.chance_node_children = np.empty(ACTION_SPACE, dtype=object)
        self.prior_probabilities = action_probabilities

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

        decision_node = self

        while decision_node.parent:

            # Since visit count was already incremented during selection for virtual loss,
            # it does not need to be incremented again during backpropagation.

            chance_node = decision_node.parent

            # Get the previous action index leading to this decision node
            prev_action_idx = decision_node.prev_action_index

            grandparent_node = chance_node.parent
            # Add Q-value and remove virtual loss added during selection
            grandparent_node.q_value_sums[prev_action_idx] += q_value + 1.0

            # Move up the tree
            decision_node = grandparent_node

    def decide_action(self, tau=1.0):
        """
        Decide on the best action to take based on the visit counts of immediate child nodes.
        Arguments:
        - tau: Temperature parameter for softmax action selection.
        This parameter modulates exploration vs exploitation in the action selection
        of the actual game.
        Returns:
        - action: The selected action proportional to the visit counts of immediate children. \
            Returns -1 if no actions are available.
        - tree_policy: The policy vector for the tree in this state, representing the
        probabilities of selecting each action derived from visit counts of the tree search.
        """

        tree_policy = np.zeros(ACTION_SPACE, dtype=np.float32)

        if self.is_terminal:
            return -1, tree_policy

        visit_counts = self.visit_counts[self.available_actions]

        if tau == 0.0:
            highest_visit_actions = self.available_actions[visit_counts == np.max(visit_counts)]
            chosen_action = np.random.choice(highest_visit_actions)
            tree_policy[chosen_action] = 1.0
            return chosen_action, tree_policy

        # Normalise visit counts using softmax with temperature tau
        visit_counts = visit_counts ** (1 / tau)
        visit_probs = visit_counts / np.sum(visit_counts)

        # Select an action based on the probabilities derived from visit counts

        chosen_action = np.random.choice(self.available_actions, p=visit_probs)

        tree_policy[self.available_actions] = visit_probs

        return chosen_action, tree_policy

    def descend(
        self,
        action_idx: int,
        tetromino_type: int
    ) -> 'MCDecisionNodeAsync':
        """
        Descend to the child node corresponding to the given action and Tetromino type
        resulting from interaction with the real environment. Detach the new root from
        its parent node. This allows reuse of the tree statistics computed from previous
        iterations

        Returns:
        - successor: The new root node of the subtree corresponding
        to the action and Tetromino type.
        """
        chance_node = self.chance_node_children[action_idx]
        successor = chance_node.decision_node_children[tetromino_type]

        # Disconnect all parent pointers (references to the unused tree) to allow garbage collection
        chance_node.parent = None
        for child in chance_node.decision_node_children:
            child.parent = None
        chance_node.decision_node_children = None

        return successor

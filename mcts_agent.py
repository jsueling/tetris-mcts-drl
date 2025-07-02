"""MCTS + DRL agent that learns how to play Tetris"""

from tqdm import tqdm

from mcts import MonteCarloTreeNode
from experience_replay_buffer import ExperienceReplayBuffer
from tetris_env import Tetris

MCTS_ITERATIONS = 800 # Number of MCTS iterations per action selection

class MCTSAgent:
    """MCTS + DRL agent for playing Tetris."""
    def __init__(self):
        self.buffer = ExperienceReplayBuffer(batch_size=128, max_size=1000000)

    def sample_replay_buffer(self):
        """Sample a batch of transitions from the replay buffer."""
        return self.buffer.sample()

    def update(self):
        pass

    def train(self, episodes=1000, tetromino_randomisation_scheme=None):
        """Run the training loop for the MCTS agent."""

        if tetromino_randomisation_scheme not in ['bag', 'uniform']:
            raise ValueError("tetromino_randomisation_scheme must be 'bag' or 'uniform'")

        env = Tetris(20, 10, tetromino_randomisation_scheme)

        for episode in tqdm(range(episodes)):

            # Create a new root node for the MCTS search tree per episode
            root_node = MonteCarloTreeNode(env)

            # Reset the environment per episode, generating the first Tetromino of the sequence
            env.reset()
            transitions = []
            done = False
            while not done:

                # TODO Finalise a state representation used for the neural network
                state = env.get_state()

                for _ in range(MCTS_ITERATIONS):
                    root_node.run_iteration()

                # The next action is decided based on the visit counts of the actions
                # available to the root node in the simulations. The tree policy is the
                # probability distribution over those actions
                action, tree_policy = root_node.decide_action(tau=1.0)

                done, current_score = env.step(action)

                # Stochastic outcome becomes deterministic by setting the next piece
                # from the actual environment.
                next_tetromino_type = env.get_next_piece()
                env.new_tetromino(next_tetromino_type)
                # Rebase the root node to the new state after the action is taken
                # and the next Tetromino is set.
                rebased_root = root_node.children[action][next_tetromino_type]
                # Remove parent node so that the search can continue from the new state
                # reusing previously computed statistics.
                rebased_root.remove_parent()
                root_node = rebased_root

                transitions.append([state, tree_policy, current_score])

            # An episode has ended, we can now compute the return-to-go (RTG)
            # by subtracting the final score from each transition's current total reward (score)
            final_score = env.score
            for t in transitions:
                t[2] = final_score - t[2]

            self.buffer.add_transitions_batch(transitions)

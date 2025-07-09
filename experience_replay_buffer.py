"""Experience replay buffer for storing and sampling transitions."""

import torch
import numpy as np

class ExperienceReplayBuffer:
    """An experience replay buffer that stores transitions and supports uniform random sampling."""
    def __init__(
        self,
        max_size: int = 1000000,
        batch_size: int = 128,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.position = 0
        self.full = False

        # The buffer will store transitions in the form of tuples:
        # 1. State will be (grid + tetromino type)
        # 2. Tree policy derived from MCTS associated with the state
        # 3. Value for the state calculated as reward-to-go when the actual game ends
        # 4. Legal actions mask for the current state

        self.states = torch.zeros((max_size, 8, 20, 10), dtype=torch.float32, device=device)
        self.tree_policies = torch.zeros((max_size, 40), dtype=torch.float32, device=device)
        self.rewards_to_go = torch.zeros((max_size,), dtype=torch.float32, device=device)
        self.legal_actions_masks = torch.zeros((max_size, 40), dtype=torch.bool, device=device)

    def add_transition(self, transition):
        """Add a single transition to the buffer."""

        self.states[self.position] = transition[0]
        self.tree_policies[self.position] = transition[1]
        self.rewards_to_go[self.position] = transition[2]
        self.legal_actions_masks[self.position] = transition[3]

        # Update position, wrapping around if necessary
        self.position = (self.position + 1) % self.max_size
        self.full = self.full or self.position == 0

    def add_transitions_batch(self, transitions):
        """
        Add batch of transitions to the buffer, making efficient use of vectorised operations
        and minimising CPU-GPU communication/transfer cost.
        """

        batch_size = len(transitions)

        states, tree_policies, rewards_to_go, legal_actions_masks = zip(*transitions)

        # Convert lists -> numpy arrays -> tensors (efficient conversion to tensors)
        states_cpu = torch.tensor(np.array(states, dtype=np.float32))
        tree_policies_cpu = torch.tensor(np.array(tree_policies, dtype=np.float32))
        rewards_to_go_cpu = torch.tensor(np.array(rewards_to_go, dtype=np.float32))
        legal_actions_masks_cpu = torch.tensor(np.array(legal_actions_masks, dtype=np.float32))

        # Move tensors to the GPU in a single operation
        states_gpu = states_cpu.to(self.device)
        tree_policies_gpu = tree_policies_cpu.to(self.device)
        rewards_to_go_gpu = rewards_to_go_cpu.to(self.device)
        legal_actions_masks_gpu = legal_actions_masks_cpu.to(self.device)

        start = self.position
        end = self.position + batch_size

        # Case 1: No wrap-around
        if end <= self.max_size:
            self.states[start:end] = states_gpu
            self.tree_policies[start:end] = tree_policies_gpu
            self.rewards_to_go[start:end] = rewards_to_go_gpu
            self.legal_actions_masks[start:end] = legal_actions_masks_gpu
            self.position = end
        # Case 2: Wrap-around
        else:

            self.full = True

            # The first segment contains elements that fill the remaining buffer capacity
            remaining_buffer_space = self.max_size - start

            self.states[start:] = states_gpu[:remaining_buffer_space]
            self.tree_policies[start:] = tree_policies_gpu[:remaining_buffer_space]
            self.rewards_to_go[start:] = rewards_to_go_gpu[:remaining_buffer_space]
            self.legal_actions_masks[start:] = legal_actions_masks_gpu[:remaining_buffer_space]

            # The second segment contains the elements that wrap around
            wrapped_end = batch_size - remaining_buffer_space

            self.states[:wrapped_end] = states_gpu[remaining_buffer_space:]
            self.tree_policies[:wrapped_end] = tree_policies_gpu[remaining_buffer_space:]
            self.rewards_to_go[:wrapped_end] = rewards_to_go_gpu[remaining_buffer_space:]
            self.legal_actions_masks[:wrapped_end] = legal_actions_masks_gpu[remaining_buffer_space:]
            self.position = wrapped_end

    def sample(self):
        """
        Uniform randomly samples a batch of transitions from the buffer.
        Returns:
            A tuple containing:
            - A batch of states (shape: [batch_size, 1, 20, 10])
            - A batch of tree policies (shape: [batch_size, 40])
            - A batch of rewards-to-go (shape: [batch_size])
            - A batch of legal actions masks (shape: [batch_size, 40])
        """

        current_size = self.max_size if self.full else self.position

        indices = torch.randint(
            0,
            current_size,
            (min(self.batch_size, current_size),),
            device=self.device
        )

        return (
            self.states[indices],
            self.tree_policies[indices],
            self.rewards_to_go[indices],
            self.legal_actions_masks[indices]
        )

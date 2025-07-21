"""Experience replay buffer for storing and sampling transitions."""

import torch

class ExperienceReplayBuffer:
    """An experience replay buffer that stores transitions and supports uniform random sampling."""
    def __init__(
        self,
        max_size: int = 500000,
        batch_size: int = 128,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.position = 0
        self.full = False

        # The buffer will store transitions in the form of tuples:
        # 1. State is (grid + tetromino type)
        # 2. Tree policy is a probability vector over the action space derived from the visit counts
        # of the tree search iterations associated with the state
        # 3. Value for the state calculated as normalised reward-to-go when the episode ends
        # 4. Legal actions mask for the current state over the action space

        self.states = torch.zeros((max_size, 8, 20, 10), dtype=torch.float32, device=device)
        self.tree_policies = torch.zeros((max_size, 40), dtype=torch.float32, device=device)
        self.normalised_rtg = torch.zeros((max_size,), dtype=torch.float32, device=device)
        self.legal_actions_masks = torch.zeros((max_size, 40), dtype=torch.bool, device=device)

    def __len__(self):
        """Return the current size of the buffer."""
        return self.max_size if self.full else self.position

    def add_transition(self, state, tree_policy, normalised_rtg, legal_actions_mask):
        """Add a single transition to the buffer."""

        self.states[self.position] = state
        self.tree_policies[self.position] = tree_policy
        self.normalised_rtg[self.position] = normalised_rtg
        self.legal_actions_masks[self.position] = legal_actions_mask

        # Update position, wrapping around if necessary
        self.position = (self.position + 1) % self.max_size
        self.full = self.full or self.position == 0

    def add_transitions_batch(self, states, tree_policies, normalised_rtg, legal_actions_masks):
        """
        Add batch of transitions to the buffer, making efficient use of vectorised operations
        and minimising CPU-GPU communication/transfer cost.
        """

        batch_size = states.shape[0]

        # Efficiently convert numpy arrays to tensors
        states_cpu = torch.from_numpy(states)
        tree_policies_cpu = torch.from_numpy(tree_policies)
        normalised_rtg_cpu = torch.from_numpy(normalised_rtg)
        legal_actions_masks_cpu = torch.from_numpy(legal_actions_masks)

        # Move tensors to the GPU in a single operation
        states_gpu = states_cpu.to(self.device)
        tree_policies_gpu = tree_policies_cpu.to(self.device)
        normalised_rtg_gpu = normalised_rtg_cpu.to(self.device)
        legal_actions_masks_gpu = legal_actions_masks_cpu.to(self.device)

        start = self.position
        end = self.position + batch_size

        # Case 1: No wrap-around
        if end <= self.max_size:
            self.states[start:end] = states_gpu
            self.tree_policies[start:end] = tree_policies_gpu
            self.normalised_rtg[start:end] = normalised_rtg_gpu
            self.legal_actions_masks[start:end] = legal_actions_masks_gpu
            self.position = end
        # Case 2: Wrap-around
        else:

            self.full = True

            # The first segment contains elements that fill the remaining buffer capacity
            remaining_buffer_space = self.max_size - start

            self.states[start:] = states_gpu[:remaining_buffer_space]
            self.tree_policies[start:] = tree_policies_gpu[:remaining_buffer_space]
            self.normalised_rtg[start:] = normalised_rtg_gpu[:remaining_buffer_space]
            self.legal_actions_masks[start:] = legal_actions_masks_gpu[:remaining_buffer_space]

            # The second segment contains the elements that wrap around
            wrapped_end = batch_size - remaining_buffer_space

            self.states[:wrapped_end] = states_gpu[remaining_buffer_space:]
            self.tree_policies[:wrapped_end] = tree_policies_gpu[remaining_buffer_space:]
            self.normalised_rtg[:wrapped_end] = normalised_rtg_gpu[remaining_buffer_space:]
            self.legal_actions_masks[:wrapped_end] = legal_actions_masks_gpu[remaining_buffer_space:]
            self.position = wrapped_end

    def sample(self):
        """
        Uniform randomly samples a batch of transitions from the buffer.
        Returns:
        - states (shape: [batch_size, 1, 20, 10])
        - tree policies (shape: [batch_size, 40])
        - normalised rewards-to-go (shape: [batch_size])
        - legal actions masks (shape: [batch_size, 40])
        """

        indices = torch.randint(
            0,
            len(self),
            (min(self.batch_size, len(self)),),
            device=self.device
        )

        return (
            self.states[indices],
            self.tree_policies[indices],
            self.normalised_rtg[indices],
            self.legal_actions_masks[indices]
        )

    def save(self, file_path):
        """Save the buffer state to a file."""

        torch.save({
            'states': self.states,
            'tree_policies': self.tree_policies,
            'normalised_rtg': self.normalised_rtg,
            'legal_actions_masks': self.legal_actions_masks,
            'position': self.position,
            'full': self.full
        }, file_path)

    def load(self, file_path):
        """Attempt to load the buffer state from a file."""

        try:
            data = torch.load(file_path, map_location=self.device)
            self.states = data['states']
            self.tree_policies = data['tree_policies']
            self.normalised_rtg = data['normalised_rtg']
            self.legal_actions_masks = data['legal_actions_masks']
            self.position = data['position']
            self.full = data['full']
        except FileNotFoundError:
            print(f"No buffer state found at {file_path}")

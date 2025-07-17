"""
Checkpointing and restoring state during training.
"""
import json
import os
import glob
import random

import numpy as np
import torch

class Checkpoint:
    """A class that handles saving and loading the entire training state of the agent"""

    def __init__(self, name, buffer, model, score_normaliser):

        directory = "../out/"
        os.makedirs(directory, exist_ok=True)
        self.out_file_prefix = directory + name
        self.buffer = buffer
        self.model = model
        self.score_normaliser = score_normaliser

    def save(self, results, hard_save=False):
        """
        Save the current state in an atomic way
        Args:
            results (dict): Dictionary containing training results to save.
            hard_save (bool): If True then saves the rng_state, buffer and model states (expensive).
        """

        # temp write
        tmp_results_file_path = self.out_file_prefix + \
            ('_tmp_hard_results.npy' if hard_save else '_tmp_results.npy')
        np.save(tmp_results_file_path, results)

        # atomic overwrite
        results_file_path = self.out_file_prefix + \
            ('_hard_results.npy' if hard_save else '_results.npy')
        os.replace(tmp_results_file_path, results_file_path)

        if not hard_save:
            return

        training_state_data = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
            "score_normalising_factor": self.score_normaliser.normalising_factor,
        }

        if torch.cuda.is_available():
            training_state_data["torch_cuda_random"] = torch.cuda.get_rng_state()

        # temp write
        tmp_state_data_file_path = self.out_file_prefix + '_tmp_state_data.npy'
        np.save(tmp_state_data_file_path, training_state_data)

        # atomic overwrite
        state_data_file_path = self.out_file_prefix + '_state_data.npy'
        os.replace(tmp_state_data_file_path, state_data_file_path)

        self.buffer.save(self.out_file_prefix)
        self.model.save(self.out_file_prefix)

    def load(self):
        """Attempt to restore state from checkpoint files and return checkpoint results."""

        # Restore training state
        try:
            training_state_file_path = self.out_file_prefix + '_state_data.npy'
            training_state = np.load(training_state_file_path, allow_pickle=True).item()

            random.setstate(training_state["python_random"])
            np.random.set_state(training_state["numpy_random"])
            torch.set_rng_state(training_state["torch_random"])

            if torch.cuda.is_available() and "torch_cuda_random" in training_state:
                torch.cuda.set_rng_state(training_state["torch_cuda_random"])

            self.score_normaliser.normalising_factor = \
                training_state["score_normalising_factor"]

        except FileNotFoundError:
            print(f"No training state found at {training_state_file_path}")

        # Restore buffer state
        self.buffer.load(self.out_file_prefix)

        # Restore model state
        self.model.load(self.out_file_prefix)

        # Restore results state and return if they exist
        try:
            results_state_file_path = self.out_file_prefix + '_hard_results.npy'
            return np.load(results_state_file_path, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"No results state found at {results_state_file_path}")
            return {}

def to_serialisable(val):
    """Convert numpy types to native Python types for JSON serialisation."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    return val

def load_and_display_results():
    """Load and print results from specified file path."""

    directory = '../out'
    pattern_a0_results = os.path.join(directory, '*_results.npy')
    results_files = glob.glob(pattern_a0_results)

    if not results_files:
        print("No results files found.")
        return

    for file in results_files:

        print("=" * 20)
        print(f"file {file}")
        print("=" * 20)

        print(
            json.dumps(
                np.load(file, allow_pickle=True).item(),
                indent=2,
                default=to_serialisable
            )
        )

        print("=" * 20)
        print("=" * 20)
        print()

if __name__ == "__main__":
    load_and_display_results()

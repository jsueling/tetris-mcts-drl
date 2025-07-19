"""
Checkpointing and restoring state during training.
"""
import json
import os
import glob
import random
import time

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

        self.results = {}

        self.episode_scores = []
        self.rolling_avg_scores = []
        self.steps_per_episode = []
        self.episode_times = []
        self.prev_step_count = self.step_count = 0
        self.starting_episode = 0

        # Attempt to restore state from checkpoint files
        self.restore_checkpoint()

    def save_episode_results(self, start_time, final_score):
        """Save the results of the episode to a file."""

        end_time = time.time()

        self.episode_scores.append(final_score)
        self.rolling_avg_scores.append(round(float(np.mean(self.episode_scores[-100:])), 2))
        self.steps_per_episode.append(self.step_count - self.prev_step_count)
        self.episode_times.append(round(float(end_time - start_time), 2))
        self.prev_step_count = self.step_count

        results = {
            "episode_scores": self.episode_scores,
            "rolling_avg_scores": self.rolling_avg_scores,
            "steps_per_episode": self.steps_per_episode,
            "episode_times": self.episode_times,
            "total_steps": self.step_count,
            "total_episodes": len(self.episode_scores),
        }

        # temp write
        tmp_results_file_path = self.out_file_prefix + '_tmp_results.npy'
        np.save(tmp_results_file_path, results)

        # atomic overwrite
        results_file_path = self.out_file_prefix + '_results.npy'
        os.replace(tmp_results_file_path, results_file_path)

    def hard_save(self):
        """Save entire training state to allow complete restoration."""

        tmp_results_file_path = self.out_file_prefix + '_tmp_hard_results.npy'
        results_file_path = self.out_file_prefix + '_hard_results.npy'
        tmp_state_data_file_path = self.out_file_prefix + '_tmp_state_data.npy'
        state_data_file_path = self.out_file_prefix + '_state_data.npy'
        tmp_model_file_path = self.out_file_prefix + '_tmp_model.pth'
        model_file_path = self.out_file_prefix + '_model.pth'
        tmp_buffer_file_path = self.out_file_prefix + "_tmp_buffer.pth"
        buffer_file_path = self.out_file_prefix + "_buffer.pth"

        training_state_data = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
            "score_normalising_factor": self.score_normaliser.normalising_factor,
        }

        if torch.cuda.is_available():
            training_state_data["torch_cuda_random"] = torch.cuda.get_rng_state()

        # Write to temporary files
        np.save(tmp_results_file_path, self.results)
        np.save(tmp_state_data_file_path, training_state_data)
        self.buffer.save(tmp_buffer_file_path)
        self.model.save(tmp_model_file_path)

        # Atomic overwrite, either the checkpoint is fully saved or not at all
        os.replace(tmp_results_file_path, results_file_path)
        os.replace(tmp_state_data_file_path, state_data_file_path)
        os.replace(tmp_model_file_path, model_file_path)
        os.replace(tmp_buffer_file_path, buffer_file_path)

    def restore_checkpoint(self):
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
        buffer_file_path = self.out_file_prefix + "_buffer.pth"
        self.buffer.load(buffer_file_path)

        # Restore model state
        model_file_path = self.out_file_prefix + '_model.pth'
        self.model.load(model_file_path)

        # Restore results state
        try:
            results_state_file_path = self.out_file_prefix + '_hard_results.npy'
            self.results = np.load(results_state_file_path, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"No results state found at {results_state_file_path}")

        self.episode_scores = self.results.get("episode_scores", [])
        self.rolling_avg_scores = self.results.get("rolling_avg_scores", [])
        self.steps_per_episode = self.results.get("steps_per_episode", [])
        self.episode_times = self.results.get("episode_times", [])
        self.prev_step_count = self.step_count = self.results.get("total_steps", 0)
        self.starting_episode = self.results.get("total_episodes", 0)

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

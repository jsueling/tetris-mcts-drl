"""
Checkpointing and restoring state during training.
"""
import json
import os
import glob
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from deep_mcts_agent import DeepMCTSAgent

class Checkpoint:
    """A class that handles saving and loading the entire training state of the agent"""

    def __init__(self, name, agent: "DeepMCTSAgent"):

        directory = "./out/"
        os.makedirs(directory, exist_ok=True)
        self.out_file_prefix = directory + name
        self.agent = agent

        # == All iterations ==

        # Episode results
        self.completed_iterations = 0
        self.benchmark_scores = []
        self.mean_score_per_episode = []
        self.mean_steps_per_episode = []
        self.mean_time_per_episode = []

        # Losses
        self.value_losses = []
        self.policy_losses = []

        # == Current iteration ==

        # Episode results
        self.episode_scores = []
        self.steps_per_episode = []
        self.episode_times = []

        # Losses
        self.current_policy_losses = []
        self.current_value_losses = []

    def log_episode_results(self, start_time, episode_score, step_count):
        """Log results of a single episode."""

        end_time = time.time()
        self.episode_scores.append(episode_score)
        self.steps_per_episode.append(step_count)
        self.episode_times.append(round(float(end_time - start_time), 2))

    def log_training_loss(self, policy_loss, value_loss):
        """Log the loss values of the current iteration."""
        self.current_policy_losses.append(round(float(policy_loss), 3))
        self.current_value_losses.append(round(float(value_loss), 3))

    def save_iteration(self, iter_idx: int):
        """
        Save entire training state per iteration to allow complete restoration.
        1. Saves latest checkpoint to a temporary file and then atomically overwrites the
        previous checkpoint file.
        2. Saves persistent checkpoints every 10 iterations.
        """

        # Iteration ends
        self.completed_iterations += 1
        self.benchmark_scores.append(self.agent.max_benchmark_score)
        self.mean_score_per_episode.append(round(float(np.mean(self.episode_scores)), 2))
        self.mean_steps_per_episode.append(round(float(np.mean(self.steps_per_episode)), 2))
        self.mean_time_per_episode.append(round(float(np.mean(self.episode_times)), 2))

        self.value_losses.append(self.current_value_losses[:])
        self.policy_losses.append(self.current_policy_losses[:])

        # Clear current iteration results
        self.episode_scores.clear()
        self.steps_per_episode.clear()
        self.episode_times.clear()

        self.current_value_losses.clear()
        self.current_policy_losses.clear()

        training_results = {
            "completed_iterations": self.completed_iterations,
            "benchmark_scores": self.benchmark_scores,
            "mean_score_per_episode": self.mean_score_per_episode,
            "mean_steps_per_episode": self.mean_steps_per_episode,
            "mean_time_per_episode": self.mean_time_per_episode,
            "value_losses": self.value_losses,
            "policy_losses": self.policy_losses,
        }

        tmp_results_file_path = self.out_file_prefix + '_tmp_results.npy'
        results_file_path = self.out_file_prefix + '_results.npy'
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
            "score_normalising_factor": self.agent.score_normaliser.normalising_factor,
        }

        if torch.cuda.is_available():
            training_state_data["torch_cuda_random"] = torch.cuda.get_rng_state()

        # Write to temporary files
        np.save(tmp_results_file_path, training_results)
        np.save(tmp_state_data_file_path, training_state_data)
        self.agent.buffer.save(tmp_buffer_file_path)
        self.agent.model.save(tmp_model_file_path)

        # Atomic overwrite, either the checkpoint is fully saved or not at all
        os.replace(tmp_results_file_path, results_file_path)
        os.replace(tmp_state_data_file_path, state_data_file_path)
        os.replace(tmp_model_file_path, model_file_path)
        os.replace(tmp_buffer_file_path, buffer_file_path)

        # Create persistent checkpoint files every 10 iterations
        if iter_idx % 10 == 0:

            iteration_identifier = f"{self.out_file_prefix}_iteration_{iter_idx}"

            persistent_training_results_fp = iteration_identifier + "_results.npy"
            persistent_state_data_fp = iteration_identifier + "_state_data.npy"
            persistent_buffer_fp = iteration_identifier + "_buffer.pth"
            persistent_model_fp = iteration_identifier + "_model.pth"

            np.save(persistent_training_results_fp, training_results)
            np.save(persistent_state_data_fp, training_state_data)
            self.agent.buffer.save(persistent_buffer_fp)
            self.agent.model.save(persistent_model_fp)

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

            self.agent.score_normaliser.normalising_factor = \
                training_state["score_normalising_factor"]

        except FileNotFoundError:
            print(f"No training state found at {training_state_file_path}")

        # Restore buffer state
        buffer_file_path = self.out_file_prefix + "_buffer.pth"
        self.agent.buffer.load(buffer_file_path)

        # Restore model state
        model_file_path = self.out_file_prefix + '_model.pth'
        if self.agent.model.load(model_file_path) is True:
            self.agent.candidate_model.load(model_file_path)

        # Restore training results state
        try:
            results_state_file_path = self.out_file_prefix + '_results.npy'
            training_results = np.load(results_state_file_path, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"No results state found at {results_state_file_path}")
            training_results = {}

        self.completed_iterations = training_results.get("completed_iterations", 0)
        self.benchmark_scores = training_results.get("benchmark_scores", [])
        self.mean_score_per_episode = training_results.get("mean_score_per_episode", [])
        self.mean_steps_per_episode = training_results.get("mean_steps_per_episode", [])
        self.mean_time_per_episode = training_results.get("mean_time_per_episode", [])
        self.value_losses = training_results.get("value_losses", [])
        self.policy_losses = training_results.get("policy_losses", [])

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

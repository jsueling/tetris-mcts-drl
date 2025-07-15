"""
Normalises unbounded Tetris scores (lines cleared) by tracking a moving average of percentiles.
This allows the agent to adapt to its own changing score distribution over time
and normalise scores to a bounded range. This implementation uses the tanh function
"""

import numpy as np

class ScoreNormaliser:
    """
    Normalises unbounded scores to a bounded range
    by keeping track of a moving average of percentile scores.
    """
    def __init__(self, alpha=0.99, percentile=95, eps=1e-5):
        self.normalising_factor = 1.0
        self.alpha = alpha
        self.percentile = percentile
        self.eps = eps

    def update(self, raw_scores):
        """Update normalising factor with batch of scores which are rewards-to-go here"""
        # Calculates the score bound at which self.percentile of scores in this batch lies below
        percentile_score_bound = np.percentile(raw_scores, self.percentile)
        # Step away from old normalising_factor towards the new percentile score bound
        self.normalising_factor = self.alpha * self.normalising_factor + \
            (1 - self.alpha) * percentile_score_bound

    def normalise(self, raw_scores):
        """Normalise scores to [-1, 1] range using adaptive normalising factor"""
        # Since raw_scores are non-negative, tanh maps them to [0, 1]
        tanh_out = np.tanh(raw_scores / (self.normalising_factor + self.eps))
        # Scale to [-1, 1] range
        return (tanh_out * 2) - 1

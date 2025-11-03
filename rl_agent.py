import numpy as np
import random
import string

class RLAgent:
    def __init__(
        self,
        alpha=0.15,             # learning rate
        gamma=0.9,              # discount factor
        epsilon=1.0,            # initial exploration rate
        epsilon_decay=0.99995,  # decay rate of epsilon
        epsilon_min=0.005,      # min exploration rate
        alpha_decay=0.9999,     # new: gradual learning rate decay
        hmm_weight=5.0,         # starting importance of HMM
        hmm_decay=0.9999,       # how fast HMM weight reduces
        hmm_min=0.1             # min HMM contribution
    ):
        """
        Reinforcement Learning Agent for Hangman with adaptive parameters.
        Combines probabilistic reasoning (HMM) with self-learned Q-values.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha_decay = alpha_decay

        # Hybrid learning weights
        self.hmm_weight = hmm_weight
        self.hmm_decay = hmm_decay
        self.hmm_min = hmm_min

        # The learned memory: Q-table
        self.Q = {}

    # -----------------------------
    # STATE ENCODING
    # -----------------------------
    def _encode_state(self, masked_word, guessed_letters):
        """Encodes game state as (pattern, guessed_count)."""
        guessed_count = len(guessed_letters)
        masked_pattern = masked_word.replace("_", "X")
        return (masked_pattern, guessed_count)

    # -----------------------------
    # ACTION SELECTION (with HMM + Q fusion)
    # -----------------------------
    def select_action(self, masked_word, guessed_letters, hmm_probs):
        """
        Choose the next letter using epsilon-greedy + adaptive HMM weighting.
        Starts probabilistic, becomes self-learned over time.
        """
        available = [ch for ch in string.ascii_lowercase if ch not in guessed_letters]

        # Random exploration (based on epsilon)
        if np.random.rand() < self.epsilon:
            action = random.choice(available)
        else:
            state = self._encode_state(masked_word, guessed_letters)
            q_vals = [self.Q.get((state, a), 0) for a in available]

            # Normalize Q-values
            if np.max(q_vals) > 0:
                q_vals = np.array(q_vals) / (np.max(q_vals) + 1e-6)
            else:
                q_vals = np.zeros(len(available))

            # Combine with HMM probabilities
            hmm_vals = np.array([hmm_probs[ord(a) - 97] for a in available])
            combined_scores = (1 - self._get_hmm_influence()) * q_vals + self._get_hmm_influence() * hmm_vals

            action = available[np.argmax(combined_scores)]

        # Decay parameters dynamically
        self._decay_parameters()

        return action

    # -----------------------------
    # Q-VALUE UPDATE
    # -----------------------------
    def update(self, state, action, reward, next_state):
        """Performs standard Q-learning update."""
        old_value = self.Q.get((state, action), 0)
        next_max = max(
            [self.Q.get((next_state, a), 0) for a in string.ascii_lowercase],
            default=0
        )
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.Q[(state, action)] = new_value

    # -----------------------------
    # INTERNAL PARAMETER DECAY FUNCTIONS
    # -----------------------------
    def _decay_parameters(self):
        """Gradually reduce epsilon, alpha, and HMM weight."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha = max(0.01, self.alpha * self.alpha_decay)
        self.hmm_weight = max(self.hmm_min, self.hmm_weight * self.hmm_decay)

    def _get_hmm_influence(self):
        """
        Converts HMM weight to a normalized influence factor between 0–1.
        Decreases as hmm_weight decays.
        """
        max_w = 5.0  # starting point
        min_w = self.hmm_min
        normalized = (self.hmm_weight - min_w) / (max_w - min_w)
        return np.clip(normalized, 0, 1)

    # -----------------------------
    # MODE CONTROL
    # -----------------------------
    def should_use_hmm(self):
        """Returns False if epsilon is very low — pure Q-mode."""
        return self.epsilon > 0.05 and self.hmm_weight > self.hmm_min + 0.05

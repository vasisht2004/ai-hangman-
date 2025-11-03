import numpy as np
import string

class HMM:
    def __init__(self, corpus_path="Data/corpus.txt"):
        self.corpus = self._load_corpus(corpus_path)
        self.global_probs = self._train(self.corpus)

    def _load_corpus(self, corpus_path):
        with open(corpus_path, "r") as f:
            words = [line.strip().lower() for line in f if line.strip().isalpha()]
        return words

    def _train(self, corpus):
        """Compute global letter probabilities (priors)."""
        letters = {ch: 1 for ch in string.ascii_lowercase}  # Laplace smoothing
        for word in corpus:
            for ch in word:
                if ch in letters:
                    letters[ch] += 1
        total = sum(letters.values())
        return {ch: letters[ch] / total for ch in letters}

    def _filter_by_pattern(self, masked_word, guessed_letters):
        """Return words from corpus matching the masked pattern."""
        filtered = []
        for word in self.corpus:
            if len(word) != len(masked_word):
                continue
            match = True
            for w_ch, m_ch in zip(word, masked_word):
                if m_ch == "_":
                    if w_ch in guessed_letters:
                        match = False
                        break
                elif w_ch != m_ch:
                    match = False
                    break
            if match:
                filtered.append(word)
        return filtered

    def get_letter_probs(self, masked_word, guessed_letters):
        """Return letter probabilities based on current masked pattern."""

        # --- Case 1: No letters revealed yet -> use global priors ---
        if all(ch == "_" for ch in masked_word):
            probs = np.array([self.global_probs[ch] for ch in string.ascii_lowercase])

        # --- Case 2: Some letters revealed -> pattern-based conditional probabilities ---
        else:
            possible_words = self._filter_by_pattern(masked_word, guessed_letters)
            letter_counts = {ch: 1 for ch in string.ascii_lowercase}  # smoothing

            if possible_words:
                for word in possible_words:
                    for i, ch in enumerate(word):
                        # only count letters that are not yet guessed and correspond to blanks
                        if masked_word[i] == "_" and ch not in guessed_letters:
                            letter_counts[ch] += 1
            else:
                # fallback to global priors if no words match
                letter_counts = {ch: self.global_probs[ch] for ch in string.ascii_lowercase}

            total = sum(letter_counts.values())
            probs = np.array([letter_counts[ch] / total for ch in string.ascii_lowercase])

        # --- Remove already guessed letters ---
        for g in guessed_letters:
            if g in string.ascii_lowercase:
                probs[ord(g) - 97] = 0

        # --- Normalize ---
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(26) / 26

        return probs

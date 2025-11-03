import os
import random

class hangman:
    def __init__(self, game_number=0):
        self.played_word = ""
        self.gameboard = []
        self.gameboard_finished = []
        self.guess = ''
        self.guess_archive = []
        self.lives = []
        self.end_state = False
        self.word_list = []
        self.game_number = game_number

        try:
            with open("Data/corpus.txt", "r") as f:
                self.word_list.extend(line.strip() for line in f)
                    
        except FileNotFoundError:
            print("Error: corpus.txt not found in Data folder.")
            self.word_list = ['default', 'backup', 'words', 'for', 'hangman']

    def reset(self):
        """Reset the game for a new round"""
        self.played_word = ""
        self.gameboard = []
        self.gameboard_finished = []
        self.guess = ''
        self.guess_archive = []
        self.lives = []
        self.end_state = False

    def set_Word(self):
        word = self.word_list[self.game_number].lower()
        self.played_word = word

    def set_finished_board(self, word):
        self.gameboard_finished = list(word)

    def set_create_board(self, word):
        self.gameboard = ['_'] * len(word)

    def set_move(self, guess, location):
        self.gameboard[location] = guess

    def set_guess(self, player_guess):
        if player_guess in self.guess_archive:
            return
        elif player_guess in self.gameboard_finished:
            for position, char in enumerate(self.gameboard_finished):
                if char == player_guess:
                    self.set_move(player_guess, position)
        else:
            self.lives.append('x')
        self.guess_archive.append(player_guess)

    def get_eg_status(self):
        if len(self.lives) >= 6 or self.gameboard == self.gameboard_finished:
            self.end_state = True

    def get_state(self):
        return {
            "masked_word": ''.join(self.gameboard),
            "guessed_letters": self.guess_archive.copy(),
            "lives_left": 6 - len(self.lives),
            "end_state": self.end_state
        }

    def step(self, guess):
        prev_correct = self.gameboard.count('_')
        self.set_guess(guess)
        self.get_eg_status()

        correct_after = self.gameboard.count('_')
        revealed_letters = prev_correct - correct_after

        done = self.end_state
        reward = 0

        if done and self.gameboard == self.gameboard_finished:
            reward = +200
        elif done:
            reward = -200
        elif revealed_letters > 0:
            reward = +50 * revealed_letters  # encourage multiple corrects
        elif guess in self.guess_archive:
            reward = -50  # penalize repeats
        else:
            reward = -10  # wrong guess

        return self.get_state(), reward, done


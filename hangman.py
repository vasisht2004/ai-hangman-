import os
import random

class hangman:
    def __init__(self):
        self.played_word = ""
        self.gameboard = []
        self.gameboard_finished = []
        self.guess = ''
        self.guess_archive = []
        self.lives = []
        self.end_state = False
        self.word_list = []

        try:
            with open("Data/corpus.txt", "r") as f:
                for i, line in enumerate(f):
                    self.word_list.append(line.strip())
                    if i == 9:
                        break
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
        word = random.choice(self.word_list)
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
        if len(self.lives) >= 6:
            self.end_state = True
        elif self.gameboard == self.gameboard_finished:
            self.end_state = True

    def get_state(self):
        return {
            "masked_word": ''.join(self.gameboard),
            "guessed_letters": self.guess_archive.copy(),
            "lives_left": 6 - len(self.lives),
            "end_state": self.end_state
        }

    def step(self, guess):
        prev_lives = len(self.lives)
        prev_board = ''.join(self.gameboard)

        self.set_guess(guess)
        self.get_eg_status()

        new_board = ''.join(self.gameboard)
        done = self.end_state

        if done and self.gameboard == self.gameboard_finished:
            reward = +100
        elif done:
            reward = -100
        elif new_board != prev_board:
            reward = +10
        elif len(self.lives) > prev_lives:
            reward = -10
        else:
            reward = -1

        return self.get_state(), reward, done

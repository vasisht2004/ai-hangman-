import os
import time
from hangman import hangman
from rl_agent import RLAgent
from hmm_model import HMM

def run_ai_games(num_games=7, sleep_time=0.8):
    hmm = HMM("Data/corpus.txt")
    agent = RLAgent(epsilon=0.1)

    for game_num in range(1, num_games + 1):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"================= GAME {game_num} =================")

        game = hangman()
        game.reset()
        game.set_Word()
        game.set_finished_board(game.played_word)
        game.set_create_board(game.played_word)

        state = game.get_state()
        done = False
        total_reward = 0

        while not done:
            masked = ''.join(game.gameboard)
            guessed = game.guess_archive.copy()
            hmm_probs = hmm.get_letter_probs(masked, guessed)

            # RL Agent picks next letter
            action = agent.select_action(masked, guessed, hmm_probs)
            next_state, reward, done = game.step(action)

            # Q-learning update
            s_key = agent._encode_state(masked, guessed)
            s_next = agent._encode_state(next_state["masked_word"], next_state["guessed_letters"])
            agent.update(s_key, action, reward, s_next)

            total_reward += reward
            state = next_state

            os.system('cls' if os.name == 'nt' else 'clear')
            print("==============================================")
            print("=                  HANGMAN                   =")
            print("==============================================")
            print("\t" + ' '.join(game.gameboard))
            print(f"  Lives: \t{''.join(game.lives)}")
            print(f"  Guesses:\t{', '.join(game.guess_archive)}")
            print("==============================================")
            print(f"Agent guessed: '{action.upper()}' | Reward: {reward}")
            time.sleep(sleep_time)

        print(f"\n✅ Game {game_num} finished! Word was: {game.played_word}")
        print(f"Total Reward: {total_reward}")
        time.sleep(1.5)

if __name__ == "__main__":
    run_ai_games(num_games=5, sleep_time=0.8)

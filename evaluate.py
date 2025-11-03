import os
import json
from hangman import hangman
from rl_agent import RLAgent
from hmm_model import HMM
import pickle


def evaluate_agent_on_test(agent, hmm, test_file="Data/test.txt", save_results=True):
    """
    Evaluate the trained RL agent + HMM on the test dataset.

    Final Score formula:
    (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)
    """

    total_words = 0
    correct_words = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0

    # Read test words
    with open(test_file, "r") as f:
        words = [line.strip().lower() for line in f.readlines() if line.strip()]

    print("\n🧠 Starting Evaluation on Test Data...\n")

    for word in words:
        total_words += 1
        game = hangman()
        game.reset()
        game.played_word = word
        game.set_finished_board(word)
        game.set_create_board(word)

        done = False
        wrong_guesses = 0
        repeated_guesses = 0

        while not done:
            masked = ''.join(game.gameboard)
            guessed = game.guess_archive.copy()
            hmm_probs = hmm.get_letter_probs(masked, guessed)

            action = agent.select_action(masked, guessed, hmm_probs)

            # Count repeated guesses
            if action in guessed:
                repeated_guesses += 1

            # Step through the game
            _, reward, done = game.step(action)

            # Count wrong guesses based on reward
            if reward == -10:
                wrong_guesses += 1

        # Check if the word was guessed correctly
        if game.gameboard == game.gameboard_finished:
            correct_words += 1
            #print(f"✅ {word.upper()}  — Guessed correctly!")
        else:
            #print(f"❌ {word.upper()}  — Failed!")
        total_wrong_guesses += wrong_guesses
        total_repeated_guesses += repeated_guesses

    # Compute metrics
    success_rate = correct_words / total_words if total_words else 0
    accuracy = success_rate * 100

    final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

    print("\n📊 FINAL EVALUATION RESULTS 📊")
    print(f"Total Words Tested:      {total_words}")
    print(f"Correctly Guessed:       {correct_words}")
    print(f"Incorrect Words:         {total_words - correct_words}")
    print(f"Accuracy:                {accuracy:.2f}%")
    print(f"Total Wrong Guesses:     {total_wrong_guesses}")
    print(f"Total Repeated Guesses:  {total_repeated_guesses}")
    print(f"🏆 Final Score:           {final_score:.2f}")

    # Save results
    if save_results:
        os.makedirs("results", exist_ok=True)
        results = {
            "total_words": total_words,
            "correct_words": correct_words,
            "incorrect_words": total_words - correct_words,
            "accuracy": accuracy,
            "total_wrong_guesses": total_wrong_guesses,
            "total_repeated_guesses": total_repeated_guesses,
            "final_score": final_score
        }
        with open("results/evaluation_summary.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\n📝 Results saved to results/evaluation_summary.json")

    return {
        "accuracy": accuracy,
        "final_score": final_score,
        "wrong_guesses": total_wrong_guesses,
        "repeats": total_repeated_guesses
    }


if __name__ == "__main__":
    # Load trained Q-table
    with open("results/trained_qtable.pkl", "rb") as f:
        q_table = pickle.load(f)

    hmm = HMM("Data/corpus.txt")
    agent = RLAgent(epsilon=0)
    agent.Q = q_table  # assign trained table

    print("\n🎯 EVALUATION MODE ACTIVE 🎯")
    evaluate_agent_on_test(agent, hmm, "Data/test.txt")

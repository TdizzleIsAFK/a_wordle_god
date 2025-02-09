# src/main.py

import argparse
import os

import torch
from data_loader import load_words, generate_training_data, WordleDataset, generate_word_files
from model import HeuristicScoringModel
from train import train_model
from gameplay import play_wordle
from torch.utils.data import DataLoader
from config import FAST_MODE
from torch.cuda import amp
from tqdm import tqdm

# Add plotting imports
import matplotlib.pyplot as plt
from collections import Counter

def main():
    scaler = amp.GradScaler()
    parser = argparse.ArgumentParser(description="Wordle Solver using ML")
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'test'], default='train', help='Mode to run the script in.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--game', type=str, help='Target word for gameplay.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Ensuring word files are generated...")
    generate_word_files('../data/words.txt', '../data/allowed_guesses.txt', '../data/possible_solutions.txt')

    print("[INFO] Loading allowed guesses and possible solutions...")
    allowed_guesses = load_words('../data/allowed_guesses.txt')
    possible_solutions = load_words('../data/possible_solutions.txt')

    input_size = 312

    if args.mode == 'train':
        print("[INFO] Generating training data...")
        training_data = generate_training_data(possible_solutions, allowed_guesses, num_games=10000)
        print(f"[INFO] Training data generated: {len(training_data)} samples")

        dataset = WordleDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

        model = HeuristicScoringModel(input_size).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("[INFO] Starting training...")
        train_model(model, dataloader, criterion, optimizer, device, scaler, args.epochs)

        torch.save(model.state_dict(), 'model.pth')
        print("[INFO] Training completed and model saved as 'model.pth'.")

    elif args.mode == 'play':
        if not args.game:
            print("[ERROR] Please provide a target word using --game")
            return
        target_word = args.game.upper()
        if len(target_word) != 5 or not target_word.isalpha():
            print("[ERROR] Please provide a valid 5-letter target word.")
            return
        if target_word not in possible_solutions:
            print("[ERROR] Target word not in the possible solutions list.")
            return

        model = HeuristicScoringModel(input_size).to(device)
        try:
            model.load_state_dict(torch.load('model.pth', map_location=device))
            model.eval()
        except FileNotFoundError:
            print("[ERROR] Model file 'model.pth' not found. Please train the model first using --mode train.")
            return

        print(f"[INFO] Starting Wordle game with target: {target_word}")
        attempts_taken = play_wordle(model, target_word, allowed_guesses, possible_solutions, device)
        print(f"[INFO] Word '{target_word}' solved in {attempts_taken} attempts.")

    elif args.mode == 'test':
        if not os.path.exists('model.pth'):
            print("[ERROR] Model file 'model.pth' not found. Please train the model first using --mode train.")
            return

        model = HeuristicScoringModel(input_size).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.eval()
        print("[INFO] Loaded trained model for testing.")

        total_attempts = 0
        total_solved = 0
        total_words = len(possible_solutions)
        attempts_list = []  # Store attempts for visualization

        print("[INFO] Starting testing over all possible solutions...")
        for target_word in tqdm(possible_solutions, desc="Testing Progress"):
            attempts = play_wordle(model, target_word, allowed_guesses, possible_solutions, device, verbose=False)
            # Record attempts (cap at 6 if not solved)
            if attempts <= 6:
                total_solved += 1
                total_attempts += attempts
                attempts_list.append(attempts)
            else:
                total_attempts += 6
                attempts_list.append(7)  # Using '7' as a marker for "not solved within 6 tries"

        average_guesses = total_attempts / total_words
        accuracy = (total_solved / total_words) * 100

        print("\n[RESULTS]")
        print(f"Total Words Tested: {total_words}")
        print(f"Words Solved: {total_solved}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Number of Guesses: {average_guesses:.2f}")

        # -----------------------------
        # Visualization
        # -----------------------------
        # 1. Histogram of attempts
        #    - Here we separate solved vs unsolved. '7' attempts is our marker for unsolved.
        solved_attempts = [a for a in attempts_list if a <= 6]
        unsolved_count = len([a for a in attempts_list if a == 7])

        plt.figure(figsize=(8, 6))
        plt.hist(solved_attempts, bins=range(1, 8), align='left', color='skyblue', edgecolor='black')
        plt.xticks(range(1, 8), ['1','2','3','4','5','6','Unsolved'])
        # Add the unsolved count on the "7" bin:
        # We'll manually place a text label instead, since '7' is our marker:
        plt.bar(7, unsolved_count, color='salmon', edgecolor='black')
        plt.xticks(range(1, 8), ['1','2','3','4','5','6','Failed'])
        plt.title("Distribution of Attempts Taken to Solve")
        plt.xlabel("Attempts")
        plt.ylabel("Number of Words")
        plt.savefig("attempt_distribution_histogram.png")
        plt.close()

        # 2. Bar chart of attempts frequency (excluding failed for clarity)
        counter = Counter(solved_attempts)
        attempts_order = [1,2,3,4,5,6]
        freq = [counter[a] for a in attempts_order]
        plt.figure(figsize=(8, 6))
        plt.bar(attempts_order, freq, color='green', edgecolor='black')
        plt.title("Frequency of Successful Attempts by Guess Count")
        plt.xlabel("Attempts")
        plt.ylabel("Number of Words Solved")
        plt.xticks(attempts_order)
        plt.savefig("successful_attempts_bar_chart.png")
        plt.close()

        print("[INFO] Visualization saved:")
        print(" - attempt_distribution_histogram.png")
        print(" - successful_attempts_bar_chart.png")


if __name__ == "__main__":
    main()

"""
Main entry point for the Wordle Solver project.
Handles training, gameplay, and testing modes.
"""

import argparse
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# Local module imports
from data_loader import load_words, generate_training_data, WordleDataset, generate_word_files, get_feedback
from model import HeuristicScoringModel
from train import train_model, evaluate_model
from gameplay import play_wordle
from config import FAST_MODE


# --- Configuration ---
DATA_DIR = os.path.join("..", "data")
ALLOWED_GUESSES_FILE = os.path.join(DATA_DIR, "allowed_guesses.txt")
POSSIBLE_SOLUTIONS_FILE = os.path.join(DATA_DIR, "possible_solutions.txt")
WORDS_FILE = os.path.join(DATA_DIR, "words.txt")
MODEL_SAVE_PATH = "model.pth"


# --- Mode-specific functions ---
def run_train(device: torch.device, args: argparse.Namespace) -> None:
    print("[INFO] Generating word files if necessary...")
    generate_word_files(WORDS_FILE, ALLOWED_GUESSES_FILE, POSSIBLE_SOLUTIONS_FILE)

    print("[INFO] Loading allowed guesses and possible solutions...")
    allowed_guesses = load_words(ALLOWED_GUESSES_FILE)
    possible_solutions = load_words(POSSIBLE_SOLUTIONS_FILE)

    print("[INFO] Generating training data...")
    training_data = generate_training_data(possible_solutions, allowed_guesses, num_games=10000)
    print(f"[INFO] Generated {len(training_data)} training samples.")

    dataset = WordleDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # Assuming input_size remains fixed
    input_size = 312
    model = HeuristicScoringModel(input_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    print("[INFO] Starting training...")
    train_model(model, dataloader, criterion, optimizer, device, scaler, args.epochs)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[INFO] Training complete. Model saved at '{MODEL_SAVE_PATH}'.")


def run_play(device: torch.device, args: argparse.Namespace) -> None:
    # Validate game argument
    if not args.game:
        print("[ERROR] Please provide a target word using --game")
        return

    target_word = args.game.upper()
    if len(target_word) != 5 or not target_word.isalpha():
        print("[ERROR] Please provide a valid 5-letter target word.")
        return

    allowed_guesses = load_words(ALLOWED_GUESSES_FILE)
    possible_solutions = load_words(POSSIBLE_SOLUTIONS_FILE)
    if target_word not in possible_solutions:
        print("[ERROR] Target word not in the possible solutions list.")
        return

    input_size = 312
    model = HeuristicScoringModel(input_size).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"[ERROR] Model file '{MODEL_SAVE_PATH}' not found. Train the model first using --mode train.")
        return

    print(f"[INFO] Starting game with target: {target_word}")
    attempts_taken = play_wordle(model, target_word, allowed_guesses, possible_solutions, device)
    print(f"[INFO] Solved '{target_word}' in {attempts_taken} attempts.")


def run_test(device: torch.device) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] Model file '{MODEL_SAVE_PATH}' not found. Train the model first using --mode train.")
        return

    allowed_guesses = load_words(ALLOWED_GUESSES_FILE)
    possible_solutions = load_words(POSSIBLE_SOLUTIONS_FILE)

    input_size = 312
    model = HeuristicScoringModel(input_size).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print("[INFO] Loaded model for testing.")

    total_attempts = 0
    total_solved = 0
    total_words = len(possible_solutions)
    attempts_list = []

    print("[INFO] Testing over all possible solutions...")
    for target_word in tqdm(possible_solutions, desc="Testing Progress"):
        attempts = play_wordle(model, target_word, allowed_guesses, possible_solutions, device, verbose=False)
        # If solved in 6 attempts or less, record; else mark as failure (7)
        if attempts <= 6:
            total_solved += 1
            total_attempts += attempts
            attempts_list.append(attempts)
        else:
            total_attempts += 6
            attempts_list.append(7)

    average_guesses = total_attempts / total_words
    accuracy = (total_solved / total_words) * 100

    print("\n[RESULTS]")
    print(f"Total Words Tested: {total_words}")
    print(f"Words Solved: {total_solved}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Guesses: {average_guesses:.2f}")


# --- Main Entry ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle Solver using ML")
    parser.add_argument(
        "--mode", type=str, choices=["train", "play", "test"], default="train",
        help="Mode to run: train, play, or test."
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--game", type=str, help="Target word for gameplay (required for 'play' mode).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if args.mode == "train":
        run_train(device, args)
    elif args.mode == "play":
        run_play(device, args)
    elif args.mode == "test":
        run_test(device)


if __name__ == "__main__":
    main()

"""
Main entry point for the Wordle Solver project.
Handles training, gameplay, and testing modes.
Now using the embedding-based model and a custom collate function.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

# Local module imports
from data_loader import (
    load_words,
    generate_training_data,
    WordleDataset,
    generate_word_files,
    get_feedback,
    collate_fn,  # Import the custom collate function.
)
from embedding_model import WordleEmbeddingModel
from train import train_model, evaluate_model
from gameplay import play_wordle
from config import FAST_MODE

# --- Configuration ---
DATA_DIR = os.path.join("..", "data")
ALLOWED_GUESSES_FILE = os.path.join(DATA_DIR, "allowed_guesses.txt")
POSSIBLE_SOLUTIONS_FILE = os.path.join(DATA_DIR, "possible_solutions.txt")
WORDS_FILE = os.path.join(DATA_DIR, "words.txt")
MODEL_SAVE_PATH = "model.pth"  # You may want to change the extension/name if needed.

# Global variables for multiprocessing workers.
GLOBAL_MODEL = None
GLOBAL_DEVICE = None
GLOBAL_ALLOWED_GUESSES = None
GLOBAL_POSSIBLE_SOLUTIONS = None


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
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,  # Use custom collate_fn to handle variable-length lists.
    )

    # Initialize the embedding-based model.
    model = WordleEmbeddingModel(letter_embedding_dim=8, mlp_hidden_dim=256).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler()

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

    model = WordleEmbeddingModel(letter_embedding_dim=8, mlp_hidden_dim=256).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"[ERROR] Model file '{MODEL_SAVE_PATH}' not found. Train the model first using --mode train.")
        return

    print(f"[INFO] Starting game with target: {target_word}")
    attempts_taken = play_wordle(model, target_word, allowed_guesses, possible_solutions, device)
    print(f"[INFO] Solved '{target_word}' in {attempts_taken} attempts.")


# --- Multiprocessing Helpers for Parallel Test Mode ---

def init_worker(model_state_path, device_str, allowed, possible):
    """
    Initializer for each worker process.
    Loads the model and sets the global variables for allowed guesses and possible solutions.
    """
    global GLOBAL_MODEL, GLOBAL_DEVICE, GLOBAL_ALLOWED_GUESSES, GLOBAL_POSSIBLE_SOLUTIONS
    import torch  # Local import for worker processes
    from embedding_model import WordleEmbeddingModel
    GLOBAL_DEVICE = torch.device(device_str)
    GLOBAL_MODEL = WordleEmbeddingModel(letter_embedding_dim=8, mlp_hidden_dim=256).to(GLOBAL_DEVICE)
    GLOBAL_MODEL.load_state_dict(torch.load(model_state_path, map_location=GLOBAL_DEVICE))
    GLOBAL_MODEL.eval()
    GLOBAL_ALLOWED_GUESSES = allowed
    GLOBAL_POSSIBLE_SOLUTIONS = possible


def simulate_game(target_word):
    """
    Worker function that simulates a single game using the globally initialized model and data.
    Returns a tuple (target_word, attempts).
    """
    from gameplay import play_wordle  # Local import to ensure module availability in worker
    attempts = play_wordle(GLOBAL_MODEL, target_word, GLOBAL_ALLOWED_GUESSES, GLOBAL_POSSIBLE_SOLUTIONS, GLOBAL_DEVICE, verbose=False)
    return (target_word, attempts)


def run_test(device: torch.device) -> None:
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] Model file '{MODEL_SAVE_PATH}' not found. Train the model first using --mode train.")
        return

    allowed_guesses = load_words(ALLOWED_GUESSES_FILE)
    possible_solutions = load_words(POSSIBLE_SOLUTIONS_FILE)

    print("[INFO] Starting parallel testing over all possible solutions...")
    total_words = len(possible_solutions)

    # Use multiprocessing to parallelize simulation of games.
    from multiprocessing import Pool, cpu_count
    num_workers = cpu_count()  # Or choose a fixed number if desired.
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(MODEL_SAVE_PATH, device.type, allowed_guesses, possible_solutions)
    ) as pool:
        # Use imap for lazy evaluation with a progress bar.
        results = list(tqdm(pool.imap(simulate_game, possible_solutions), total=total_words, desc="Testing Progress"))

    total_attempts = 0
    total_solved = 0
    attempts_list = []

    for target_word, attempts in results:
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
    parser = argparse.ArgumentParser(description="Wordle Solver using ML (Embedding-based Model)")
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
    # Set the multiprocessing start method to "spawn" to work with CUDA.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()

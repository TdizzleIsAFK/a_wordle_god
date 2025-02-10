"""
Main entry point for the Wordle Solver project.
Handles training, gameplay, and testing modes.
Now using the embedding-based model, a cached training data mechanism,
and a custom collate function.
"""

import argparse
import os
import pickle
import random
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool

# Local module imports
from data_loader import (
    load_words,
    generate_training_data,
    WordleDataset,
    generate_word_files,
    get_feedback,
    collate_fn,  # Custom collate function.
)
from embedding_model import WordleEmbeddingModel
from train import train_model, evaluate_model
from gameplay import play_wordle
from config import FAST_MODE

# --- Configuration ---
DATA_DIR = os.path.join("..", "data")
ALLOWED_GUESSES_FILE = os.path.join(DATA_DIR, "all_allowed_guesses.txt")
POSSIBLE_SOLUTIONS_FILE = os.path.join(DATA_DIR, "all_possible_solutions.txt")
WORDS_FILE = os.path.join(DATA_DIR, "all_words.txt")
MODEL_SAVE_PATH = "model.pth"  # Adjust as needed.
TRAINING_DATA_PKL = "training_data.pkl"  # Cache file for training data.

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

    # Check if cached training data exists; force regeneration if --force is provided.
    if args.force or not os.path.exists(TRAINING_DATA_PKL):
        print("[INFO] Generating training data...")
        training_data = generate_training_data(possible_solutions, allowed_guesses, num_games=10000)
        print(f"[INFO] Generated {len(training_data)} training samples.")
        with open(TRAINING_DATA_PKL, "wb") as f:
            pickle.dump(training_data, f)
    else:
        print("[INFO] Loading cached training data...")
        with open(TRAINING_DATA_PKL, "rb") as f:
            training_data = pickle.load(f)

    dataset = WordleDataset(training_data)
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0,       # Use 0 workers if you run into shared-memory issues.
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Initialize the embedding-based model.
    model = WordleEmbeddingModel(letter_embedding_dim=16, mlp_hidden_dim=512).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler()

    # --- New code for resuming from a checkpoint ---
    start_epoch = 0
    checkpoint_files = glob.glob("model_*.pth")
    latest_checkpoint = None
    latest_epoch = -1
    for file in checkpoint_files:
        try:
            # Expect filenames like "model_400.pth"
            epoch_num = int(file.split("_")[-1].split(".")[0])
            # Use <= so that if we have a checkpoint with epoch equal to target, we pick it.
            if epoch_num > latest_epoch and epoch_num <= args.epochs:
                latest_epoch = epoch_num
                latest_checkpoint = file
        except Exception as e:
            pass

    if latest_checkpoint is not None:
        # If we already have a checkpoint at or beyond the target, assume training is complete.
        if latest_epoch == args.epochs:
            print(f"[INFO] Found checkpoint from epoch {latest_epoch}. Training is already complete.")
            return
        else:
            print(f"[INFO] Loading checkpoint from {latest_checkpoint} (epoch {latest_epoch})")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            else:
                # If the checkpoint is just a state_dict.
                model.load_state_dict(checkpoint)
                print("[WARN] Checkpoint does not contain optimizer and scaler states. Resuming training without them.")
            start_epoch = latest_epoch
    else:
        print("[INFO] No checkpoint found; training from scratch.")
    # --- End new code for resuming ---

    total_epochs = args.epochs
    print(f"[INFO] Training from epoch {start_epoch} to {total_epochs}...")
    train_model(model, dataloader, criterion, optimizer, device, scaler, total_epochs, start_epoch=start_epoch)

    # Call train_model with the starting epoch (we will modify train_model to accept this)
    train_model(model, dataloader, criterion, optimizer, device, scaler, total_epochs, start_epoch=start_epoch)

    # Save the checkpoint with the total epoch count as part of the filename.
    checkpoint_file = f"model_{total_epochs}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }, checkpoint_file)
    print(f"[INFO] Training complete. Model saved at '{checkpoint_file}'.")

def run_play(device: torch.device, args: argparse.Namespace) -> None:
    # Validate game argument.
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

    model = WordleEmbeddingModel(letter_embedding_dim=16, mlp_hidden_dim=512).to(device)
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
    import torch  # Local import in worker.
    from embedding_model import WordleEmbeddingModel

    GLOBAL_DEVICE = torch.device(device_str)
    GLOBAL_MODEL = WordleEmbeddingModel(letter_embedding_dim=16, mlp_hidden_dim=512).to(GLOBAL_DEVICE)

    # Load the checkpoint.
    checkpoint = torch.load(model_state_path, map_location=GLOBAL_DEVICE)
    # If the checkpoint is a dictionary with a key 'model_state_dict', extract it.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    GLOBAL_MODEL.load_state_dict(checkpoint)
    GLOBAL_MODEL.eval()

    GLOBAL_ALLOWED_GUESSES = allowed
    GLOBAL_POSSIBLE_SOLUTIONS = possible



def simulate_game(target_word):
    """
    Worker function that simulates a single game using the globally initialized model and data.
    Returns a tuple (target_word, attempts).
    """
    from gameplay import play_wordle  # Ensure module availability in worker.
    attempts = play_wordle(GLOBAL_MODEL, target_word, GLOBAL_ALLOWED_GUESSES, GLOBAL_POSSIBLE_SOLUTIONS, GLOBAL_DEVICE, verbose=False)
    return (target_word, attempts)

def run_test(device: torch.device) -> None:
    # Look for all checkpoint files following the naming convention "model_*.pth"
    checkpoint_files = glob.glob("model_*.pth")
    if not checkpoint_files:
        print(f"[ERROR] No checkpoint file found. Train the model first using --mode train.")
        return

    # Find the checkpoint with the highest epoch number.
    latest_checkpoint = None
    latest_epoch = -1
    for file in checkpoint_files:
        try:
            # Expect filenames like "model_400.pth"
            epoch_num = int(file.split("_")[-1].split(".")[0])
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = file
        except Exception:
            pass

    if latest_checkpoint is None:
        print("[ERROR] No valid checkpoint found.")
        return

    print(f"[INFO] Using checkpoint: {latest_checkpoint} (epoch {latest_epoch})")

    allowed_guesses = load_words(ALLOWED_GUESSES_FILE)
    possible_solutions = load_words(POSSIBLE_SOLUTIONS_FILE)

    # Take a random sample of 1000 words (if there are at least 1000 words)
    if len(possible_solutions) > 1000:
        sampled_solutions = random.sample(possible_solutions, 1000)
    else:
        sampled_solutions = possible_solutions

    total_words = len(sampled_solutions)
    print(f"[INFO] Starting parallel testing over a random sample of {total_words} words...")

    # Use a Pool of workers and pass the latest checkpoint to the initializer.
    num_workers = 8  # Adjust upward if your GPU can handle more concurrency.
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(latest_checkpoint, device.type, allowed_guesses, possible_solutions)
    ) as pool:
        results = list(tqdm(pool.imap(simulate_game, sampled_solutions), total=total_words, desc="Testing Progress"))

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
    parser.add_argument("--force", action="store_true", help="Force regeneration of cached training data.")
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
    # Use the spawn start method so that CUDA can be used in child processes.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    # Set the sharing strategy to 'file_system' to avoid mmap issues.
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    main()
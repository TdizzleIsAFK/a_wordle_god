# data_loader.py

import random
from typing import List, Tuple
from constraints import WordleConstraints
from torch.utils.data import Dataset
import torch
import os
from config import FAST_MODE
import pickle  # <-- Import pickle for caching

from embedding_encoding import (
    encode_word_indices,
    encode_constraints_indices,
    encode_presence_set,
    encode_absent_set,
)


def load_words(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        words = [line.strip().upper() for line in f if len(line.strip()) == 5 and line.strip().isalpha()]
    if not FAST_MODE:
        print(f"[INFO] Loaded {len(words)} words from {file_path}.")
    return words


def get_feedback(guess: str, target: str) -> List[str]:
    feedback = ['B'] * 5
    target_letters = list(target)

    for i in range(5):  # Greens
        if guess[i] == target[i]:
            feedback[i] = 'G'
            target_letters[i] = None

    for i in range(5):  # Yellows
        if feedback[i] == 'B' and guess[i] in target_letters:
            feedback[i] = 'Y'
            target_letters[target_letters.index(guess[i])] = None

    return feedback


def generate_word_files(words_file: str, guesses_file: str, solutions_file: str):
    if not FAST_MODE:
        print("[INFO] Checking if word files need to be generated...")
    if not os.path.exists(guesses_file) or not os.path.exists(solutions_file):
        if not FAST_MODE:
            print("[INFO] Generating allowed guesses and possible solutions files...")
        if not os.path.exists(words_file):
            raise FileNotFoundError(f"[ERROR] The words file '{words_file}' does not exist.")

        with open(words_file, 'r') as words:
            all_words = [line.strip().upper() for line in words if len(line.strip()) == 5 and line.strip().isalpha()]

        if not FAST_MODE:
            print(f"[INFO] Total words loaded from {words_file}: {len(all_words)}")

        if not all_words:
            raise ValueError("[ERROR] The words.txt file is empty or improperly formatted.")

        with open(guesses_file, 'w') as guesses:
            guesses.write('\n'.join(all_words))
        if not FAST_MODE:
            print(f"[INFO] {len(all_words)} words written to {guesses_file}.")

        subset_size = len(all_words)  # Use the entire vocabulary as possible solutions.
        with open(solutions_file, 'w') as solutions:
            solutions.write('\n'.join(all_words[:subset_size]))
        if not FAST_MODE:
            print(f"[INFO] {subset_size} words written to {solutions_file}.")

    else:
        if not FAST_MODE:
            print("[INFO] Allowed guesses and possible solutions files already exist. Skipping generation.")


def generate_training_data(possible_solutions: List[str], allowed_guesses: List[str], num_games: int = 2000):
    if not FAST_MODE:
        print("[INFO] Generating synthetic training games...")
    data = []
    for game_idx in range(num_games):
        target = random.choice(possible_solutions)
        constraints = WordleConstraints()
        current_possible = possible_solutions.copy()
        used_letters = set()
        used_guesses = set()

        # **First Attempt: Always "CRANE"**
        first_guess = "CRANE"
        if first_guess not in allowed_guesses:
            raise ValueError("[ERROR] 'CRANE' must be in the allowed_guesses list.")

        guess = first_guess
        used_guesses.add(guess)
        feedback = get_feedback(guess, target)
        if not FAST_MODE:
            print(f"[INFO] Game {game_idx + 1}: First Guess = {guess}, Feedback = {feedback}")

        constraints.update_constraints(guess, feedback)
        current_possible = constraints.filter_words(current_possible)
        label = len(current_possible)
        data.append((constraints, guess, label))
        used_letters.update(guess)

        if guess == target:
            continue  # Game solved in first attempt

        # **Subsequent Attempts: 2 to 5**
        for attempt in range(1, 5):
            if attempt < 3:
                valid_guesses = []
                for word in allowed_guesses:
                    letters = set(word)
                    new_letters = letters - used_letters
                    if all(word.count(letter) <= 1 for letter in new_letters):
                        if word not in used_guesses:
                            valid_guesses.append(word)
            else:
                valid_guesses = [w for w in allowed_guesses if w not in used_guesses]

            if not valid_guesses:
                break  # No valid guesses left

            guess = random.choice(valid_guesses)
            used_guesses.add(guess)
            feedback = get_feedback(guess, target)
            constraints.update_constraints(guess, feedback)
            current_possible = constraints.filter_words(current_possible)
            label = len(current_possible)
            data.append((constraints, guess, label))
            used_letters.update(guess)

            if guess == target:
                break  # Game solved

    if not FAST_MODE:
        print(f"[INFO] Synthetic training data generation complete. Total samples: {len(data)}")
    return data


def load_or_generate_training_data(possible_solutions: List[str], allowed_guesses: List[str],
                                   num_games: int = 2000, cache_file: str = "training_data.pkl"):
    """
    Checks if a cached training data file exists.
    If yes, loads the training samples from disk.
    Otherwise, generates the training data and caches it.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            training_data = pickle.load(f)
        print(f"[INFO] Loaded cached training data from {cache_file}.")
        return training_data
    else:
        print(f"[INFO] Cache file {cache_file} not found. Generating training data...")
        training_data = generate_training_data(possible_solutions, allowed_guesses, num_games)
        with open(cache_file, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"[INFO] Training data generated and cached in {cache_file}.")
        return training_data


class WordleDataset(Dataset):
    def __init__(self, data: List[Tuple[WordleConstraints, str, int]]):
        """
        Each sample is now a tuple:
          (guess_indices, constraint_indices, presence, absent, label)
        where:
            - guess_indices: Tensor of shape (5,) for the candidate guess word (0-25).
            - constraint_indices: Tensor of shape (5,) for known correct letters (0-25, with 26 for unknown).
            - presence: List of indices for letters known to be present.
            - absent: List of indices for letters known to be absent.
            - label: The numerical label (number of possible solutions left).
        """
        self.samples = []
        for constraints, word, label in data:
            guess_indices = encode_word_indices(word)  # shape: (5,)
            constraint_indices = encode_constraints_indices(constraints.correct_positions)
            presence = encode_presence_set(constraints.present_letters)
            absent = encode_absent_set(constraints.absent_letters)
            self.samples.append((guess_indices, constraint_indices, presence, absent, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Custom collate function to handle batching of samples with variable-length presence and absent lists.

    Returns:
      - guess_indices: Tensor of shape (batch_size, 5)
      - constraint_indices: Tensor of shape (batch_size, 5)
      - presence_list: List of length batch_size, each element is a list of indices
      - absent_list: List of length batch_size, each element is a list of indices
      - labels: Tensor of shape (batch_size, 1)
    """
    guess_indices = torch.stack([item[0] for item in batch], dim=0)
    constraint_indices = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(1)
    presence_list = [item[2] for item in batch]
    absent_list = [item[3] for item in batch]
    return guess_indices, constraint_indices, presence_list, absent_list, labels

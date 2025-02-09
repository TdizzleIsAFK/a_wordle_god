# src/data_loader.py

import random
from typing import List, Tuple
from constraints import WordleConstraints
from encoding import encode_input
from torch.utils.data import Dataset
import torch
import os
from config import FAST_MODE

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

        subset_size = min(2500, len(all_words))
        with open(solutions_file, 'w') as solutions:
            solutions.write('\n'.join(all_words[:subset_size]))
        if not FAST_MODE:
            print(f"[INFO] {subset_size} words written to {solutions_file}.")
    else:
        if not FAST_MODE:
            print("[INFO] Allowed guesses and possible solutions files already exist. Skipping generation.")

# src/data_loader.py

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
            print(f"[INFO] Game {game_idx+1}: First Guess = {guess}, Feedback = {feedback}")

        constraints.update_constraints(guess, feedback)
        current_possible = constraints.filter_words(current_possible)
        label = len(current_possible)
        data.append((constraints, guess, label))
        used_letters.update(guess)

        if guess == target:
            continue  # Game solved in first attempt

        # **Subsequent Attempts: 2 to 5**
        for attempt in range(1, 5):
            # Apply the new rule: For the first three attempts (including first), restrict new letters
            if attempt < 3:
                # Identify new letters that haven't been used yet
                valid_guesses = []
                for word in allowed_guesses:
                    letters = set(word)
                    new_letters = letters - used_letters
                    # Ensure that no new letter is repeated more than once
                    if all(word.count(letter) <= 1 for letter in new_letters):
                        # Additionally, you can exclude "CRANE" if it's already used
                        if word not in used_guesses:
                            valid_guesses.append(word)
            else:
                # After attempt #3, allow any guesses excluding used guesses
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

class WordleDataset(Dataset):
    def __init__(self, data: List[Tuple[WordleConstraints, str, int]]):
        self.inputs = []
        self.labels = []
        for constraints, word, label in data:
            input_tensor = encode_input(constraints, word)
            # We trust our data now, no debug print necessary.
            self.inputs.append(input_tensor)
            self.labels.append(label)
        self.inputs = torch.stack(self.inputs)  # Shape: (N, 312)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

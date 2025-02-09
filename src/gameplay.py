# src/gameplay.py

import torch
from model import HeuristicScoringModel
from constraints import WordleConstraints
from encoding import encode_input
from typing import List, Dict
from data_loader import get_feedback
from config import FAST_MODE

def encode_batch_constraints(constraints: WordleConstraints, words: List[str]) -> torch.Tensor:
    batch_size = len(words)
    correct_pos = torch.zeros(batch_size, 5 * 26)
    present = torch.zeros(batch_size, 26)
    absent = torch.zeros(batch_size, 26)

    for i, word in enumerate(words):
        for pos, letter in enumerate(constraints.correct_positions):
            if letter:
                index = ord(letter) - ord('A')
                correct_pos[i, pos * 26 + index] = 1
        for letter in constraints.present_letters:
            index = ord(letter) - ord('A')
            present[i, index] = 1
        for letter in constraints.absent_letters:
            index = ord(letter) - ord('A')
            absent[i, index] = 1

    return torch.cat([correct_pos, present, absent], dim=1)

def encode_batch_words(words: List[str]) -> torch.Tensor:
    batch_size = len(words)
    encoding = torch.zeros(batch_size, 5 * 26)
    for i, word in enumerate(words):
        for pos, letter in enumerate(word):
            index = ord(letter) - ord('A')
            encoding[i, pos * 26 + index] = 1
    return encoding


def calculate_entropy_for_guess(guess: str, constraints: WordleConstraints, possible_solutions: List[str]) -> float:
    """
    Calculate the expected size of the solution set after making this guess.
    We do this by:
    - For each possible solution, determine the feedback that guess would produce.
    - Group solutions by that feedback.
    - For each feedback pattern, we know the resulting filtered solution set size.
    - Compute the weighted average (expected value) of the solution set size after receiving that feedback.
    Lower is better (more information gain).

    Instead of formal entropy, we focus on expected reduction in candidates.
    You can modify the metric as needed, but expected solution count is a good proxy.
    """
    # Dictionary: feedback_pattern -> count of solutions that yield that pattern
    feedback_groups: Dict[tuple, int] = {}
    # We'll store a representative constraints update for each feedback pattern
    # to know how many words remain after that feedback.
    # Actually, we can just count first, then filter.

    # First, identify all feedback patterns
    from copy import deepcopy
    for sol in possible_solutions:
        fb = tuple(get_feedback(guess, sol))
        feedback_groups[fb] = feedback_groups.get(fb, 0) + 1

    # Now, for each feedback pattern, apply it to constraints and determine how many remain.
    # To do this properly, we simulate the updated constraints.
    total_solutions = len(possible_solutions)
    expected_remaining = 0.0

    # We'll compute how the constraints would look after this guess and feedback.
    for fb, count in feedback_groups.items():
        new_constraints = deepcopy(constraints)
        new_constraints.update_constraints(guess, list(fb))
        # Filter possible solutions under these new constraints
        filtered = new_constraints.filter_words(possible_solutions)
        remaining = len(filtered)
        # Probability of this feedback pattern
        p = count / total_solutions
        expected_remaining += p * remaining

    return expected_remaining


def select_best_guess_entropy(constraints: WordleConstraints, allowed_guesses: List[str], possible_solutions: List[str]) -> str:
    """
    Select the guess that yields the lowest expected number of remaining solutions.
    This should maximize information gain.
    """
    best_guess = None
    best_score = float('inf')

    # If the possible_solutions list is small, it's more efficient to just choose from them.
    # Otherwise, consider all allowed_guesses.
    candidates = allowed_guesses

    for guess in candidates:
        # Skip guesses that violate known constraints trivially
        if not constraints.is_word_possible(guess):
            continue

        # Calculate expected remaining solutions after this guess
        score = calculate_entropy_for_guess(guess, constraints, possible_solutions)

        if score < best_score:
            best_score = score
            best_guess = guess

    return best_guess if best_guess is not None else allowed_guesses[0]


def play_wordle(model: HeuristicScoringModel, target: str, allowed_guesses: List[str], possible_solutions: List[str], device: torch.device, verbose: bool = True):
    constraints = WordleConstraints()
    current_possible = possible_solutions.copy()
    used_letters = set()
    attempts = 0
    used_guesses = set()

    # **First Attempt: Always "CRANE"**
    first_guess = "CRANE"
    if first_guess not in allowed_guesses:
        raise ValueError("[ERROR] 'CRANE' must be in the allowed_guesses list.")

    guess = first_guess
    used_guesses.add(guess)
    feedback = get_feedback(guess, target)

    if verbose and not FAST_MODE:
        print(f"[INFO] Attempt {attempts+1}: Guess = {guess}, Feedback = {feedback}")

    if guess == target:
        if verbose and not FAST_MODE:
            print("[INFO] Solved!")
        return attempts + 1

    constraints.update_constraints(guess, feedback)
    current_possible = constraints.filter_words(current_possible)
    used_letters.update(guess)
    attempts += 1

    # **Subsequent Attempts: 2 to 5**
    while attempts < 6:
        if attempts < 3:
            # Apply your existing rules
            valid_guesses = []
            for word in allowed_guesses:
                if word in used_guesses:
                    continue
                letters = set(word)
                new_letters = letters - used_letters
                # Ensure that no new letter is repeated more than once
                if all(word.count(letter) <= 1 for letter in new_letters):
                    valid_guesses.append(word)
        else:
            # After attempt #3, you allow any unused guesses
            valid_guesses = [w for w in allowed_guesses if w not in used_guesses]

        # NOW filter by constraints:
        valid_guesses = constraints.filter_words(valid_guesses)

        if not valid_guesses:
            if verbose and not FAST_MODE:
                print("[DEBUG] No valid guesses remaining.")
            break

        # **Use Entropy-based Guess Selection**
        guess = select_best_guess_entropy(constraints, valid_guesses, current_possible)
        used_guesses.add(guess)

        feedback = get_feedback(guess, target)
        if verbose and not FAST_MODE:
            print(f"[INFO] Attempt {attempts+1}: Guess = {guess}, Feedback = {feedback}")

        if guess == target:
            if verbose and not FAST_MODE:
                print("[INFO] Solved!")
            return attempts + 1

        constraints.update_constraints(guess, feedback)
        current_possible = constraints.filter_words(current_possible)
        used_letters.update(guess)
        attempts += 1

    if verbose and not FAST_MODE:
        print(f"[INFO] Failed to solve. The word was: {target}")
    return attempts + 1

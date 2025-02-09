"""
Module for gameplay routines including encoding functions and entropy-based guess selection.
"""

from copy import deepcopy
from typing import List, Dict, Tuple
import torch

from model import HeuristicScoringModel
from constraints import WordleConstraints
from data_loader import get_feedback
from config import FAST_MODE


def encode_batch_constraints(constraints: WordleConstraints, words: List[str]) -> torch.Tensor:
    """
    Encode constraint data for a batch of words.
    - correct_pos: one-hot for correct positions (5 positions × 26 letters)
    - present and absent: 26-dimensional one-hot vectors
    """
    batch_size = len(words)
    correct_pos = torch.zeros(batch_size, 5 * 26)
    present = torch.zeros(batch_size, 26)
    absent = torch.zeros(batch_size, 26)

    for i, _ in enumerate(words):
        for pos, letter in enumerate(constraints.correct_positions):
            if letter:
                correct_pos[i, pos * 26 + (ord(letter) - ord('A'))] = 1
        for letter in constraints.present_letters:
            present[i, ord(letter) - ord('A')] = 1
        for letter in constraints.absent_letters:
            absent[i, ord(letter) - ord('A')] = 1

    return torch.cat([correct_pos, present, absent], dim=1)


def encode_batch_words(words: List[str]) -> torch.Tensor:
    """
    Encode a list of words as one-hot vectors.
    Each word is represented as a 5x26 flattened one-hot encoding.
    """
    batch_size = len(words)
    encoding = torch.zeros(batch_size, 5 * 26)
    for i, word in enumerate(words):
        for pos, letter in enumerate(word):
            encoding[i, pos * 26 + (ord(letter) - ord('A'))] = 1
    return encoding


def calculate_entropy_for_guess(guess: str, constraints: WordleConstraints, possible_solutions: List[str]) -> float:
    """
    Calculate the expected remaining number of solutions after making a guess.
    For each possible feedback pattern (from the guess against all possible solutions):
      - Update constraints and filter the solution set.
      - Compute the weighted average of the remaining candidates.
    Lower expected counts imply higher information gain.
    """
    feedback_groups: Dict[Tuple[str, ...], int] = {}
    total_solutions = len(possible_solutions)

    for sol in possible_solutions:
        fb = tuple(get_feedback(guess, sol))
        feedback_groups[fb] = feedback_groups.get(fb, 0) + 1

    expected_remaining = 0.0
    for fb, count in feedback_groups.items():
        new_constraints = deepcopy(constraints)
        new_constraints.update_constraints(guess, list(fb))
        filtered = new_constraints.filter_words(possible_solutions)
        remaining = len(filtered)
        p = count / total_solutions
        expected_remaining += p * remaining

    return expected_remaining


def select_best_guess_entropy(constraints: WordleConstraints, allowed_guesses: List[str], possible_solutions: List[str]) -> str:
    """
    Select the guess from allowed_guesses that minimizes the expected number of remaining solutions.
    This heuristic maximizes information gain.
    """
    best_guess = None
    best_score = float('inf')

    for guess in allowed_guesses:
        if not constraints.is_word_possible(guess):
            continue
        score = calculate_entropy_for_guess(guess, constraints, possible_solutions)
        if score < best_score:
            best_score = score
            best_guess = guess

    return best_guess if best_guess is not None else allowed_guesses[0]


def play_wordle(model: HeuristicScoringModel, target: str, allowed_guesses: List[str],
                possible_solutions: List[str], device: torch.device, verbose: bool = True) -> int:
    """
    Simulate a Wordle game using the ML model and constraints.
    Uses a fixed first guess ("CRANE") and then selects subsequent guesses based on an entropy heuristic.
    Returns the number of attempts taken.
    """
    constraints = WordleConstraints()
    current_possible = possible_solutions.copy()
    used_letters = set()
    attempts = 0
    used_guesses = set()

    # First guess: Fixed strategy
    first_guess = "CRANE"
    if first_guess not in allowed_guesses:
        raise ValueError("[ERROR] 'CRANE' must be in the allowed guesses list.")

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

    # Subsequent attempts (up to 6 total)
    while attempts < 6:
        # Early attempts: Favor guesses with few repeating new letters.
        if attempts < 3:
            valid_guesses = [
                word for word in allowed_guesses
                if word not in used_guesses and
                   all(word.count(letter) <= 1 for letter in set(word) - used_letters)
            ]
        else:
            valid_guesses = [word for word in allowed_guesses if word not in used_guesses]

        valid_guesses = constraints.filter_words(valid_guesses)
        if not valid_guesses:
            if verbose and not FAST_MODE:
                print("[DEBUG] No valid guesses remaining.")
            break

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

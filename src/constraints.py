# src/constraints.py
from typing import List, Set, Dict
from config import FAST_MODE

class WordleConstraints:
    def __init__(self):
        self.correct_positions = [None] * 5
        self.present_letters: Set[str] = set()
        self.absent_letters: Set[str] = set()
        # New data structure to track letters not allowed in certain positions
        # This maps each letter to a set of positions it cannot occupy.
        self.letter_not_in_position: Dict[str, Set[int]] = {chr(c): set() for c in range(ord('A'), ord('Z')+1)}

    def update_constraints(self, guess: str, feedback: List[str]):
        # Before updating, keep track of which letters we know are in the solution
        # (either from correct_positions or present_letters)
        known_present_letters = set(l for l in self.correct_positions if l is not None) | self.present_letters

        for i, (g, f) in enumerate(zip(guess, feedback)):
            if f == 'G':
                # Letter is correct in this position
                self.correct_positions[i] = g
                # If a letter is correct, it should not be marked absent or restricted at this position
                if g in self.absent_letters:
                    self.absent_letters.remove(g)
                if g in self.present_letters:
                    self.present_letters.remove(g)
                # Since this position is correct, we don't need to mark it in letter_not_in_position
                # as it is correct here.
            elif f == 'Y':
                # Letter is present in the word, but not here
                # Only add to present_letters if it's not already a correct position letter
                if g not in self.correct_positions:
                    self.present_letters.add(g)
                # Mark that this letter cannot appear in this position
                self.letter_not_in_position[g].add(i)
            elif f == 'B':
                # Letter is absent at this position.
                # However, if we already know the letter is present elsewhere (from Y or G in previous guesses),
                # this means the letter isn't absent altogether, it's just not allowed in this position.
                # If we don't know that the letter is present, then it's truly absent.
                if g not in self.correct_positions and g not in self.present_letters:
                    # Truly absent letter
                    self.absent_letters.add(g)
                else:
                    # We know this letter is present from a previous guess, so it can't be at this position.
                    self.letter_not_in_position[g].add(i)

    def is_word_possible(self, word: str) -> bool:
        # Check correct positions
        for i, letter in enumerate(self.correct_positions):
            if letter and word[i] != letter:
                return False

        # Check that all present letters appear at least once in the word
        for letter in self.present_letters:
            if letter not in word:
                return False

        # Check that no absent letters appear in the word
        for letter in self.absent_letters:
            if letter in word:
                return False

        # Check positional restrictions for known present letters
        # If a letter is known not to be in a certain position, ensure this word doesn't place it there.
        for i, w_letter in enumerate(word):
            if i in self.letter_not_in_position[w_letter]:
                return False

        return True

    def filter_words(self, words: List[str]) -> List[str]:
        return [word for word in words if self.is_word_possible(word)]

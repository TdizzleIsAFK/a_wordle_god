"""
Module for managing Wordle game constraints.
"""

from typing import List, Set, Dict, Optional
from config import FAST_MODE  # Assuming FAST_MODE is defined in config.py


class WordleConstraints:
    def __init__(self) -> None:
        # Correct letters in their exact positions (None if unknown)
        self.correct_positions: List[Optional[str]] = [None] * 5
        # Letters known to be present but not yet correctly placed
        self.present_letters: Set[str] = set()
        # Letters confirmed absent from the target word
        self.absent_letters: Set[str] = set()
        # Maps each letter (A-Z) to a set of positions where it cannot appear
        self.letter_not_in_position: Dict[str, Set[int]] = {chr(c): set() for c in range(ord('A'), ord('Z') + 1)}

    def update_constraints(self, guess: str, feedback: List[str]) -> None:
        """
        Update the constraints based on a guess and its feedback.
        Feedback codes:
         - 'G': Correct letter at correct position.
         - 'Y': Letter exists but in a different position.
         - 'B': Letter absent from the word (unless known elsewhere).
        """
        known_present = {l for l in self.correct_positions if l is not None} | self.present_letters

        for i, (g, f) in enumerate(zip(guess, feedback)):
            if f == 'G':
                self.correct_positions[i] = g
                # Remove from absent/present if already there
                self.absent_letters.discard(g)
                self.present_letters.discard(g)
            elif f == 'Y':
                if g not in self.correct_positions:
                    self.present_letters.add(g)
                self.letter_not_in_position[g].add(i)
            elif f == 'B':
                # If we already know the letter exists somewhere else, just mark the position
                if g not in known_present:
                    self.absent_letters.add(g)
                else:
                    self.letter_not_in_position[g].add(i)

    def is_word_possible(self, word: str) -> bool:
        """
        Determine if a given word is compatible with the current constraints.
        """
        # Check exact positions
        for i, letter in enumerate(self.correct_positions):
            if letter and word[i] != letter:
                return False
        # Must contain all letters known to be present
        if any(letter not in word for letter in self.present_letters):
            return False
        # Should not contain any letters known to be absent
        if any(letter in word for letter in self.absent_letters):
            return False
        # Check letter position restrictions
        for i, w_letter in enumerate(word):
            if i in self.letter_not_in_position[w_letter]:
                return False
        return True

    def filter_words(self, words: List[str]) -> List[str]:
        """
        Filter a list of words, returning only those that satisfy the constraints.
        """
        return [word for word in words if self.is_word_possible(word)]

# src/encoding.py
import torch
from constraints import WordleConstraints
from typing import List

def encode_constraints(constraints: WordleConstraints) -> torch.Tensor:
    correct_pos = torch.zeros(5 * 26)
    present = torch.zeros(26)
    absent = torch.zeros(26)

    for i, letter in enumerate(constraints.correct_positions):
        if letter:
            index = ord(letter) - ord('A')
            correct_pos[i * 26 + index] = 1

    for letter in constraints.present_letters:
        index = ord(letter) - ord('A')
        present[index] = 1

    for letter in constraints.absent_letters:
        index = ord(letter) - ord('A')
        absent[index] = 1

    return torch.cat([correct_pos, present, absent])

def encode_word(word: str) -> torch.Tensor:
    encoding = torch.zeros(5 * 26)
    for i, letter in enumerate(word):
        index = ord(letter) - ord('A')
        encoding[i * 26 + index] = 1
    return encoding

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

def encode_input(constraints: WordleConstraints, word: str) -> torch.Tensor:
    return torch.cat([encode_constraints(constraints), encode_word(word)])

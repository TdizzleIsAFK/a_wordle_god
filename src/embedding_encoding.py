# embedding_encoding.py
import torch

def encode_word_indices(word: str) -> torch.Tensor:
    """
    Encodes a 5-letter word into a tensor of letter indices (0-25).
    """
    if len(word) != 5:
        raise ValueError("Word must be 5 letters long.")
    indices = [ord(letter.upper()) - ord('A') for letter in word]
    return torch.tensor(indices, dtype=torch.long)

def encode_constraints_indices(constraints: list) -> torch.Tensor:
    """
    Encodes the constraint letters into a tensor of indices.
    For unknown letters (None or empty), returns 26.
    Expects a list of length 5.
    """
    if len(constraints) != 5:
        raise ValueError("Constraints must be a list of 5 elements.")
    indices = []
    for letter in constraints:
        if letter is None or letter == '':
            indices.append(26)
        else:
            indices.append(ord(letter.upper()) - ord('A'))
    return torch.tensor(indices, dtype=torch.long)

def encode_presence_set(presence_set: set) -> list:
    """
    Encodes a set of letters known to be present into a list of indices.
    """
    return [ord(letter.upper()) - ord('A') for letter in presence_set]

def encode_absent_set(absent_set: set) -> list:
    """
    Encodes a set of letters known to be absent into a list of indices.
    """
    return [ord(letter.upper()) - ord('A') for letter in absent_set]

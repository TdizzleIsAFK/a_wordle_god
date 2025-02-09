# embedding_encoding.py
import torch


def encode_word_indices(word: str) -> torch.Tensor:
    """
    Encodes a 5-letter word into a tensor of integer indices.
    Each letter A–Z is mapped to 0–25.

    Args:
        word: A 5‑letter string.
    Returns:
        Tensor of shape (5,) with dtype=torch.long.
    """
    if len(word) != 5:
        raise ValueError("Word must be exactly 5 letters.")
    indices = [ord(letter.upper()) - ord('A') for letter in word]
    return torch.tensor(indices, dtype=torch.long)


def encode_constraints_indices(correct_positions: list) -> torch.Tensor:
    """
    Encodes the constraint for correct positions into a tensor of indices.
    For each position, if a letter is known (A–Z), use its index (0–25); if unknown,
    use index 26.

    Args:
        correct_positions: A list of 5 elements, where each element is an uppercase letter
                           or an empty string/None if unknown.
    Returns:
        Tensor of shape (5,) with dtype=torch.long.
    """
    if len(correct_positions) != 5:
        raise ValueError("Correct positions list must have 5 elements.")
    indices = []
    for letter in correct_positions:
        if letter is None or letter == '':
            indices.append(26)
        else:
            indices.append(ord(letter.upper()) - ord('A'))
    return torch.tensor(indices, dtype=torch.long)


def encode_presence_set(present_letters: list) -> list:
    """
    Encodes a list of present letters into a list of indices.

    Args:
        present_letters: A list of uppercase letters.
    Returns:
        A list of integers.
    """
    return [ord(letter.upper()) - ord('A') for letter in present_letters]


def encode_absent_set(absent_letters: list) -> list:
    """
    Encodes a list of absent letters into a list of indices.

    Args:
        absent_letters: A list of uppercase letters.
    Returns:
        A list of integers.
    """
    return [ord(letter.upper()) - ord('A') for letter in absent_letters]

# embedding_model.py
import torch
import torch.nn as nn


class WordleEmbeddingModel(nn.Module):
    def __init__(self, letter_embedding_dim=16, mlp_hidden_dim=512):
        """
        Args:
            letter_embedding_dim: Dimension for the learned letter embeddings.
            mlp_hidden_dim: Number of hidden units in the MLP that follows.
        """
        super(WordleEmbeddingModel, self).__init__()
        self.letter_embedding_dim = letter_embedding_dim

        # Embedding for candidate guess letters (indices 0-25)
        self.guess_embedding = nn.Embedding(num_embeddings=26, embedding_dim=letter_embedding_dim)
        # Embedding for constraint letters (0-25 for known letters; use index 26 for unknown)
        self.constraint_embedding = nn.Embedding(num_embeddings=27, embedding_dim=letter_embedding_dim)

        # Total features:
        # - 5 letters from candidate guess: 5 * letter_embedding_dim
        # - 5 letters from constraints: 5 * letter_embedding_dim
        # - Aggregated embedding for presence set: letter_embedding_dim
        # - Aggregated embedding for absent set: letter_embedding_dim
        total_dim = (5 + 5) * letter_embedding_dim + 2 * letter_embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def vectorized_aggregate(self, letter_lists):
        """
        Aggregates a batch of variable-length letter index lists into a tensor of shape
        (batch_size, letter_embedding_dim) by averaging the embeddings in a vectorized manner.

        Args:
            letter_lists: List (length = batch_size) where each element is a list of letter indices (0–25).
        Returns:
            A tensor of shape (batch_size, letter_embedding_dim) with the averaged embeddings.
        """
        device = self.guess_embedding.weight.device
        dtype = self.guess_embedding.weight.dtype
        batch_size = len(letter_lists)

        # Determine the maximum length among the lists; if all are empty, return zeros.
        max_len = max((len(lst) for lst in letter_lists), default=0)
        if max_len == 0:
            return torch.zeros(batch_size, self.letter_embedding_dim, device=device, dtype=dtype)

        # Pad each list with a pad-value (-1) so that each sample has length max_len.
        # The pad value (-1) is chosen because it is outside the valid range [0, 25].
        padded = [lst + [-1] * (max_len - len(lst)) for lst in letter_lists]
        # Create a tensor on the target device.
        padded_tensor = torch.tensor(padded, dtype=torch.long, device=device)  # Shape: (B, max_len)

        # Create a mask indicating valid positions (1.0 for valid, 0.0 for pad).
        mask = (padded_tensor != -1).unsqueeze(-1).to(dtype)  # Shape: (B, max_len, 1)

        # For padded positions, clamp the indices to 0 (they will be zeroed-out by the mask).
        padded_tensor_clamped = padded_tensor.clamp(min=0)

        # Lookup embeddings in batch: shape (B, max_len, letter_embedding_dim)
        embedded = self.guess_embedding(padded_tensor_clamped)
        # Zero out the padded positions.
        embedded = embedded * mask

        # Sum over the time dimension and compute the number of valid tokens.
        sum_emb = embedded.sum(dim=1)  # Shape: (B, letter_embedding_dim)
        counts = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero.
        mean_emb = sum_emb / counts

        # (Optional) For any samples that originally had no valid tokens, mean_emb will be 0.
        return mean_emb

    def forward(self, guess_indices, constraint_indices, presence_list, absent_list):
        """
        Forward pass.
        Args:
            guess_indices: Tensor of shape (batch_size, 5) with candidate guess letter indices (0–25).
            constraint_indices: Tensor of shape (batch_size, 5) with known correct letter indices (0–25)
                                or 26 for unknown positions.
            presence_list: List (length=batch_size) where each element is a list of letter indices (0–25)
                           for letters known to be present.
            absent_list: List (length=batch_size) where each element is a list of letter indices (0–25)
                         for letters known to be absent.
        Returns:
            Tensor of shape (batch_size, 1) representing the model’s output score.
        """
        batch_size = guess_indices.size(0)

        # Candidate guess embeddings: (batch_size, 5, emb_dim) → flatten to (batch_size, 5*emb_dim)
        guess_emb = self.guess_embedding(guess_indices)
        guess_emb_flat = guess_emb.view(batch_size, -1)

        # Constraint embeddings: (batch_size, 5, emb_dim) → flatten to (batch_size, 5*emb_dim)
        constraint_emb = self.constraint_embedding(constraint_indices)
        constraint_emb_flat = constraint_emb.view(batch_size, -1)

        # Vectorized aggregation for the presence and absent letter sets.
        presence_agg = self.vectorized_aggregate(presence_list)  # Shape: (batch_size, emb_dim)
        absent_agg = self.vectorized_aggregate(absent_list)  # Shape: (batch_size, emb_dim)

        # Concatenate all features.
        combined = torch.cat([guess_emb_flat, constraint_emb_flat, presence_agg, absent_agg], dim=1)
        out = self.mlp(combined)
        return out

# embedding_model.py
import torch
import torch.nn as nn


class WordleEmbeddingModel(nn.Module):
    def __init__(self, letter_embedding_dim=8, mlp_hidden_dim=256):
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

    def aggregate_set(self, letter_set):
        """
        Aggregates a list of letter indices (e.g., letters known to be present or absent)
        into a single embedding by averaging.
        Args:
            letter_set: A Python list of integer indices (each 0-25).
        Returns:
            A tensor of shape (letter_embedding_dim,).
        """
        if len(letter_set) == 0:
            # If the set is empty, return a zero vector.
            return torch.zeros(self.letter_embedding_dim, device=self.guess_embedding.weight.device)
        indices = torch.tensor(letter_set, dtype=torch.long, device=self.guess_embedding.weight.device)
        embedded = self.guess_embedding(indices)  # Shape: (num_letters, letter_embedding_dim)
        return embedded.mean(dim=0)

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

        # For each sample in the batch, aggregate presence and absent letter embeddings.
        presence_agg = []
        absent_agg = []
        for i in range(batch_size):
            pres_emb = self.aggregate_set(presence_list[i])
            abs_emb = self.aggregate_set(absent_list[i])
            presence_agg.append(pres_emb)
            absent_agg.append(abs_emb)
        presence_agg = torch.stack(presence_agg, dim=0)  # Shape: (batch_size, emb_dim)
        absent_agg = torch.stack(absent_agg, dim=0)  # Shape: (batch_size, emb_dim)

        # Concatenate all features.
        combined = torch.cat([guess_emb_flat, constraint_emb_flat, presence_agg, absent_agg], dim=1)
        out = self.mlp(combined)
        return out

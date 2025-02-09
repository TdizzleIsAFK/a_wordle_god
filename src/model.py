# src/model.py
import torch
import torch.nn as nn


class HeuristicScoringModel(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_hidden_layers=3, dropout=0.3):
        super(HeuristicScoringModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

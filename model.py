# Import relevant libraries and dependencies

import torch
import torch.nn as nn


# Single-layer LSTM architecture
class MyLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_layers):
        super(MyLSTM, self).__init__()
        # LSTM parameters
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers

        # Layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size + 1)  # vocab_size + 1 ('T' - term. symbol)
        self.sigmoid = nn.Sigmoid()

    # Initialize the hidden and cell states of the LSTM with zeros.
    def init_hidden(self, batch_size=1):
        return [torch.zeros(self.hidden_layers, batch_size, self.hidden_dim) for _ in (1, 2)]

    def forward(self, inputs, hidden0):
        output, hidden = self.lstm(inputs, hidden0)  # Apply the LSTM layer
        output = self.linear(output)  # Apply the linear layer
        output = self.sigmoid(output)  # Apply the sigmoid layer
        return output, hidden

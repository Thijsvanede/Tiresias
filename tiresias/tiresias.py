from arrayLSTM import ArrayLSTM, StochasticArrayLSTM
from lstm import LSTM
from module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiresias(Module):

    def __init__(self, size_input, size_hidden, size_output, k):
        # Initialise super
        super().__init__()

        # Set dimensions
        self.size_input  = size_input
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.k           = k

        # Initialise layers
        self.lstm    = nn.LSTM(size_input, size_hidden, batch_first=True)
        # self.lstm    = LSTM(size_input, size_hidden)
        # self.lstm    = ArrayLSTM(size_input, size_hidden, k)
        self.linear  = nn.Linear(size_hidden, size_output)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # One-hot encode input
        encoded = F.one_hot(X, self.size_input).to(torch.float32)

        # Pass through LSTM layer
        out, (hidden, state) = self.lstm(encoded)
        # Take hidden state as output
        hidden = hidden.squeeze(0)

        # Pass through linear layer
        out = self.linear(hidden)
        # Perform softmax and return
        return self.softmax(out)

    def predict(self, X, variable=False):

        result = super().predict(X, variable=variable)
        topv, topi = result.topk(1)
        topi = topi.reshape(-1)
        return topi

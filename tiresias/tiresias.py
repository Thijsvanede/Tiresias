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
        self.lstm    = nn.LSTM  (size_input , size_hidden, batch_first=True)
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

        result = super().predict(X, variable)
        topv, topi = result.topk(1)
        topi = topi.reshape(-1)
        return topi

if __name__ == "__main__":
    import torch
    import numpy as np
    X = [[1, 2, 3, 4],
         [1, 2, 3, 4, 5],
         [1, 2],
         [1, 2, 3, 4, 5, 6],
         [6, 2, 3, 4, 5, 6],
         [6, 2, 3, 4],
         [6, 2, 3],
         [6, 2, 3, 4, 5],
         [6, 2, 3, 4, 5],
         [2, 2, 3, 4, 5],
         [5, 2, 3, 4, 5]]

    y = torch.as_tensor([1, 1, 1, 1, 6, 6, 6, 6, 6, 1, 2, 5])

    X_train = X[:int(len(X)*.8) ]
    y_train = y[:int(len(y)*.8) ]
    X_test  = X[ int(len(X)*.8):]
    y_test  = y[ int(len(y)*.8):]

    tiresias = Tiresias(10, 128, 10)
    tiresias.fit(X_train, y_train, epochs=100, batch_size=1, variable=True)
    y_pred = tiresias.predict(X_test, variable=True)

    print(y_pred)
    print(y_test)
    print(y_pred == y_test)

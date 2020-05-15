from arrayLSTM import ArrayLSTM, SoftArrayLSTM, StochasticArrayLSTM
from lstm import LSTM
from module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiresias(Module):
    """Implementation of Tiresias

        From `Tiresias: Predicting security events through deep learning`_ by Shen et al.

        .. _`Tiresias: Predicting security events through deep learning`: https://doi.org/10.1145/3243734.3243811

        Note
        ----
        This is a `batch_first=True` implementation, hence the `forward()`
        method expect inputs of `shape=(batch, seq_len, input_size)`.

        Attributes
        ----------
        input_size : int
            Size of input dimension

        hidden_size : int
            Size of hidden dimension

        output_size : int
            Size of output dimension

        k : int
            Number of parallel memory structures, i.e. cell states to use
        """

    def __init__(self, input_size, hidden_size, output_size, k):
        """Implementation of Tiresias

            Parameters
            ----------
            input_size : int
                Size of input dimension

            hidden_size : int
                Size of hidden dimension

            output_size : int
                Size of output dimension

            k : int
                Number of parallel memory structures, i.e. cell states to use
            """
        # Initialise super
        super().__init__()

        # Set dimensions
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k           = k

        # Initialise layers
        # self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.lstm    = LSTM(input_size, hidden_size)
        # self.lstm    = ArrayLSTM(input_size, hidden_size, k)
        self.lstm    = SoftArrayLSTM(input_size, hidden_size, k, max_pooling=True)
        # self.lstm    = StochasticArrayLSTM(input_size, hidden_size, k)
        self.linear  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # One-hot encode input
        encoded = F.one_hot(X, self.input_size).to(torch.float32)

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

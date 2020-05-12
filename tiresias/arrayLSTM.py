from lstm import LSTM

import torch
import torch.nn as nn

class ArrayLSTM(LSTM):

    def __init__(self, input_size, hidden_size, k):
        """Implementation of ArrayLSTM """
        # Call super
        super().__init__(input_size, hidden_size)

        # Set dimensions
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.k           = k

        # Set parameters
        self.W_f = nn.Parameter(torch.Tensor(k, input_size , hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(k, hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(k, hidden_size))

        self.W_i = nn.Parameter(torch.Tensor(k, input_size , hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(k, hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(k, hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(k, input_size , hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(k, hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(k, hidden_size))

        self.W_c_ = nn.Parameter(torch.Tensor(k, input_size , hidden_size))
        self.U_c_ = nn.Parameter(torch.Tensor(k, hidden_size, hidden_size))
        self.b_c_ = nn.Parameter(torch.Tensor(k, hidden_size))

        # Initialise weights
        self.init_weights()

    ########################################################################
    #                         Pass through network                         #
    ########################################################################

    def _forward_single_(self, x, h_t, c_t):
        """Perform a single forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor of shape=(batch, input_size)
                Tensor to pass through network

            h_i : torch.Tensor of shape (batch, input_size)
                Tensor containing the hidden state

            c_i : torch.Tensor of shape (batch, input_size)
                Tensor containing the cell state

            Returns
            -------
            h_i : torch.Tensor of shape (batch, input_size)
                Tensor containing the next hidden state

            c_i : torch.Tensor of shape (k, batch, input_size)
                Tensor containing the next cell state
            """
        # Initialise result
        o = torch.Tensor(self.k, *x.shape)
        c = torch.Tensor(self.k, *x.shape)

        # Pass through network
        for k in range(self.k):
            f_t_k   = torch.sigmoid(x @ self.W_f [k] + h_t @ self.U_f [k] + self.b_f [k])
            i_t_k   = torch.sigmoid(x @ self.W_i [k] + h_t @ self.U_i [k] + self.b_i [k])
            o_t_k   = torch.sigmoid(x @ self.W_o [k] + h_t @ self.U_o [k] + self.b_o [k])
            c_t_k_  = torch.tanh   (x @ self.W_c_[k] + h_t @ self.U_c_[k] + self.b_c_[k])
            c_t[k]  = f_t_k * c_t[k] + i_t_k * c_t_k_

            # Update variables
            o[k] =o_t_k
            c[k] =c_t[k]

        # Update h
        h_t = self.update_h(o, c)

        # Return result
        return h_t, c_t

    ########################################################################
    #                         Update hidden state                          #
    ########################################################################

    def update_h(self, o, c):
        """Default hidden state as sum of o_k and c_k"""
        # Initialise h
        h = torch.zeros(o[0].shape)

        # Loop over all outputs
        for k in range(self.k):
            # Increment h
            h += o[k] * torch.tanh(c[k])

        # Return result
        return h

    ########################################################################
    #                     Hidden state initialisation                      #
    ########################################################################

    def initHidden(self, x):
        """Initialise hidden layer"""
        return torch.zeros(        x.shape[0], self.hidden_size).to(x.device),\
               torch.zeros(self.k, x.shape[0], self.hidden_size).to(x.device)


class StochasticArrayLSTM(ArrayLSTM):

    def update_h(self, o, c):
        """Update hidden state based on most likely output of o"""
        # Compute probability
        probability = nn.functional.softmax(o.sum(dim=2), dim=0)
        # Get top probabilities
        topv, topi = probability.topk(1, dim=0)
        # Remove dimension
        topi = topi.squeeze(0)

        # Select most likely items
        o_i = o[topi, torch.arange(o.shape[1])]
        c_i = c[topi, torch.arange(c.shape[1])]

        # Compute h
        h = o_i * torch.tanh(c_i)

        # Return result
        return h


if __name__ == "__main__":

    # alstm = ArrayLSTM(32, 32, 10)
    alstm = StochasticArrayLSTM(32, 32, 10)
    # alstm = LSTM(32, 32)
    x = torch.Tensor(1024, 5, 32)
    x = nn.init.xavier_uniform_(x)

    y, (h_t, o_t) = alstm(x)
    print(y.shape, h_t.shape, o_t.shape)

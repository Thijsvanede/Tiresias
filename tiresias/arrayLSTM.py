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

        # Set layers
        self.i2h = nn.Linear(input_size , 4*hidden_size*k)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size*k)

    ########################################################################
    #                         Pass through network                         #
    ########################################################################

    def _forward_single_(self, x, hidden, state):
        """Perform a single forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor of shape=(batch, input_size)
                Tensor to pass through network

            hidden : torch.Tensor of shape (batch, input_size)
                Tensor containing the hidden state

            state : torch.Tensor of shape (batch, input_size)
                Tensor containing the cell state

            Returns
            -------
            hidden : torch.Tensor of shape (batch, input_size)
                Tensor containing the next hidden state

            state : torch.Tensor of shape (batch, input_size)
                Tensor containing the next cell state
            """
        # Reshape hidden state to work for single cell
        hidden = hidden.view(hidden.size(1), -1)
        # Initialise outputs
        outputs = torch.zeros(self.k, x.shape[0], self.hidden_size)

        # Apply linear mapping
        linear = self.i2h(x) + self.h2h(hidden)
        # View linear in terms of k
        linear = linear.view(x.shape[0], self.k, -1)

        # Loop over all k
        for k, linear_ in enumerate(torch.unbind(linear, dim=1)):
            # Perform activation functions
            gates = linear_[:, :3*self.hidden_size ].sigmoid()
            c_t   = linear_[:,  3*self.hidden_size:].tanh()

            # Extract gates
            f_t = gates[:, :self.hidden_size                   ]
            i_t = gates[:,  self.hidden_size:2*self.hidden_size]
            o_t = gates[:, -self.hidden_size:                  ]

            # Update state
            state[k] = torch.mul(state[k].clone(), f_t) + torch.mul(i_t, c_t)
            # Update outputs
            outputs[k] = o_t

        # Update hidden state
        hidden = self.update_hidden(outputs, state)

        # Return result
        return hidden, state

    ########################################################################
    #                         Update hidden state                          #
    ########################################################################

    def update_hidden(self, outputs, states):
        """Default hidden state as sum of outputs and cells

            Parameters
            ----------
            outputs : torch.Tensor of shape=(k, batch_size, hidden_size)
                Tensor containing the result of output gates o

            states : torch.Tensor of shape=(k, batch_size, hidden_size)
                Tensor containing the cell states

            Returns
            -------
            hidden : torch.Tensor of shape=(1, batch_size, hidden_size)
                Hidden tensor as computed from outputs and states
            """
        # Initialise hidden state
        hidden = torch.zeros(1, outputs.shape[1], self.hidden_size)

        # Loop over all outputs
        for output, state in zip(torch.unbind(outputs, dim=0),
                                 torch.unbind(states , dim=0)):
            # Update hidden state
            hidden += torch.mul(output, state.tanh())

        # Return hiddens tate
        return hidden

    ########################################################################
    #                     Hidden state initialisation                      #
    ########################################################################

    def initHidden(self, x):
        """Initialise hidden layer"""
        return torch.zeros(     1, x.shape[0], self.hidden_size).to(x.device),\
               torch.zeros(self.k, x.shape[0], self.hidden_size).to(x.device)


class StochasticArrayLSTM(ArrayLSTM):

    def update_hidden(self, o, c):
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

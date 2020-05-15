import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        """Regular LSTM implementation in pytorch

            Parameters
            ----------
            input_size : int
                Size of input dimension

            hidden_size : int
                Size of hidden dimension
            """
        # Call super
        super().__init__()

        # Set dimensions
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Set layers
        self.i2h = nn.Linear(input_size , 4*hidden_size)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size)


    def forward(self, x, hidden=None):
        """Forward all sequences through the network.

            Parameters
            ----------
            x : torch.Tensor of shape=(batch, seq_len, input_size)
                Tensor to pass through network

            hidden : torch.Tensor of shape (batch, input_size), default=0 vector
                Tensor containing the hidden state

            state : torch.Tensor of shape (batch, input_size), default=0 vector
                Tensor containing the cell state
            """
        # Initialise hidden state if necessary
        hidden, state = hidden or self.initHidden(x)

        # Initialise outputs
        outputs = list()

        # Loop over all timesteps
        for x_ in torch.unbind(x, dim=1):
            # Perform a single forward pass
            hidden, state = self._forward_single_(x_, hidden, state)
            # Append output
            outputs.append(hidden[0].clone())

        # Return result
        return torch.stack(outputs, dim=1), (hidden, state)


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
        # Reshape to work for single cell
        hidden = hidden.view(hidden.size(1), -1)
        state  = state .view(state .size(1), -1)

        # Linear mapping
        linear = self.i2h(x) + self.h2h(hidden)

        # Perform activation functions
        gates = linear[:, :3*self.hidden_size ].sigmoid()
        c_t_  = linear[:,  3*self.hidden_size:].tanh()

        # Extract gates
        f_t = gates[:, :self.hidden_size                   ]
        i_t = gates[:,  self.hidden_size:2*self.hidden_size]
        o_t = gates[:, -self.hidden_size:                  ]

        # Update states
        state  = torch.mul(state, f_t) + torch.mul(i_t, c_t_)
        hidden = torch.mul(o_t, state.tanh())

        # Reshape to work for single cell
        hidden = hidden.view(1, hidden.size(0), -1)
        state  = state .view(1, state .size(0), -1)

        # Return result
        return hidden, state

    ########################################################################
    #                     Hidden state initialisation                      #
    ########################################################################

    def initHidden(self, x):
        """Initialise hidden layer"""
        return torch.zeros(1, x.shape[0], self.hidden_size).to(x.device),\
               torch.zeros(1, x.shape[0], self.hidden_size).to(x.device)

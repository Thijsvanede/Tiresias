# Import pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import pytorch LSTM implementation
from lstm import LSTM

class ArrayLSTM(LSTM):
    """Implementation of ArrayLSTM

        From `Recurrent Memory Array Structures`_ by Kamil Rocki

        .. _`Recurrent Memory Array Structures`: https://arxiv.org/abs/1607.03085

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

        k : int
            Number of parallel memory structures, i.e. cell states to use

        i2h : nn.Linear
            Linear layer transforming input to hidden state

        h2h : nn.Linear
            Linear layer updating hidden state to hidden state
        """

    def __init__(self, input_size, hidden_size, k):
        """Implementation of ArrayLSTM

            Note
            ----
            This is a `batch_first=True` implementation, hence the `forward()`
            method expect inputs of `shape=(batch, seq_len, input_size)`.

            Parameters
            ----------
            input_size : int
                Size of input dimension

            hidden_size : int
                Size of hidden dimension

            k : int
                Number of parallel memory structures, i.e. cell states to use
            """
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
        outputs = torch.zeros(self.k, x.shape[0], self.hidden_size, device=x.device)

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
        hidden = torch.zeros(1, outputs.shape[1], self.hidden_size, device=states.device)

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








class SoftArrayLSTM(ArrayLSTM):
    """Implementation of ArrayLSTM with Lane selection: Soft attention

        From `Recurrent Memory Array Structures`_ by Kamil Rocki

        .. _`Recurrent Memory Array Structures`: https://arxiv.org/abs/1607.03085

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

        k : int
            Number of parallel memory structures, i.e. cell states to use

        i2h : nn.Linear
            Linear layer transforming input to hidden state

        h2h : nn.Linear
            Linear layer updating hidden state to hidden state
        """

    def __init__(self, input_size, hidden_size, k):
        """Implementation of ArrayLSTM

            Note
            ----
            This is a `batch_first=True` implementation, hence the `forward()`
            method expect inputs of `shape=(batch, seq_len, input_size)`.

            Parameters
            ----------
            input_size : int
                Size of input dimension

            hidden_size : int
                Size of hidden dimension

            k : int
                Number of parallel memory structures, i.e. cell states to use
            """
        # Call super
        super().__init__(input_size, hidden_size, k)

        # Set attention layer
        self.i2a = nn.Linear(input_size, hidden_size*k)

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

        # Compute attention signal
        attention = self.i2a(x).sigmoid()
        # View attention in terms of k
        attention = attention.view(x.shape[0], self.k, -1)
        # Compute softmax s
        softmax = F.softmax(attention, dim=1)

        # Apply linear mapping
        linear = self.i2h(x) + self.h2h(hidden)
        # View linear in terms of k
        linear = linear.view(x.shape[0], self.k, -1)

        # Loop over all k
        for k, (linear_, softmax_) in enumerate(zip(
                                        torch.unbind(linear , dim=1),
                                        torch.unbind(softmax, dim=1))):
            # Perform activation functions
            gates = linear_[:, :3*self.hidden_size ].sigmoid()
            c_t   = linear_[:,  3*self.hidden_size:].tanh()

            # Extract gates
            f_t = torch.mul(softmax_, gates[:, :self.hidden_size                   ])
            i_t = torch.mul(softmax_, gates[:,  self.hidden_size:2*self.hidden_size])
            o_t = torch.mul(softmax_, gates[:, -self.hidden_size:                  ])

            # Update state
            state[k] = torch.mul(state[k].clone(), (1-f_t)) + torch.mul(i_t, c_t)
            # Update outputs
            outputs[k] = o_t

        # Update hidden state
        hidden = self.update_hidden(outputs, state)

        # Return result
        return hidden, state











class StochasticArrayLSTM(ArrayLSTM):

    def update_hidden(self, outputs, states):
        """Update hidden state based on most likely output

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
        # Compute probability
        probability = nn.functional.softmax(outputs.sum(dim=2), dim=0)
        # Get top probabilities
        topv, topi = probability.topk(1, dim=0)
        # Remove dimension
        topi = topi.squeeze(0)

        # Select most likely items
        output_i = outputs[topi, torch.arange(outputs.shape[1])]
        state_i  = states [topi, torch.arange(states .shape[1])]

        # Compute h
        hidden = output_i * torch.tanh(state_i)
        # View hidden properly
        hidden = hidden.unsqueeze(0)

        # Return result
        return hidden

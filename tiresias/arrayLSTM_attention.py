# Import pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import standard arrayLSTM implementation
from arrayLSTM import ArrayLSTM

class AttentionArrayLSTM(ArrayLSTM):
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

        max_pooling : boolean, default=False
            If True, uses max pooling for attention instead

        i2h : nn.Linear
            Linear layer transforming input to hidden state

        h2h : nn.Linear
            Linear layer updating hidden state to hidden state
        """

    def __init__(self, input_size, hidden_size, k, max_pooling=False):
        """Implementation of ArrayLSTM with Lane selection: Soft attention

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

            max_pooling : boolean, default=False
                If True, uses max pooling for attention instead
            """
        # Call super
        super().__init__(input_size, hidden_size, k)

        # Set max_pooling
        self.max_pooling = max_pooling

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
        outputs = torch.zeros(self.k, x.shape[0], self.hidden_size, device=x.device)

        # Compute attention signal
        attention = self.i2a(x).sigmoid()
        # View attention in terms of k
        attention = attention.view(x.shape[0], self.k, -1)
        # Compute softmax s
        softmax = F.softmax(attention, dim=1)
        # Use max_pooling if necessary
        if self.max_pooling:
            # Get maximum
            softmax = softmax.max(dim=1).values
            softmax = softmax.unsqueeze_(1)
            softmax = softmax.expand(-1, self.k, -1)

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

import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        """Regular LSTM implementation

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

        # Set parameters
        self.W_f = nn.Parameter(torch.Tensor(input_size , hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_i = nn.Parameter(torch.Tensor(input_size , hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(input_size , hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_c_ = nn.Parameter(torch.Tensor(input_size , hidden_size))
        self.U_c_ = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c_ = nn.Parameter(torch.Tensor(hidden_size))

        # Initialise weights
        self.init_weights()

    ########################################################################
    #                         Pass through network                         #
    ########################################################################

    def forward(self, x, hidden=None):
        """Forward all sequences through the network.

            Parameters
            ----------
            x : torch.Tensor of shape=(batch, seq_len, input_size)
                Tensor to pass through network

            h_i : torch.Tensor of shape (batch, input_size), default=0 vector
                Tensor containing the hidden state

            c_i : torch.Tensor of shape (batch, input_size), default=0 vector
                Tensor containing the cell state
            """
        # Read shape
        batch, seq_len, input_size = x.shape
        # Initialise result
        result = list()

        # Initialise/unpack hidden states
        h_t, c_t = hidden or self.initHidden(x)

        # Iterate over timesteps
        for t in range(seq_len):
            # Extract relevant timestep
            x_t = x[:, t, :]
            # Feed through network
            h_t, c_t = self._forward_single_(x_t, h_t, c_t)
            # Append result
            result.append(h_t.unsqueeze(0))

        # Get as array
        result = torch.cat(result).transpose(0, 1).contiguous()

        # Return result
        return result, (h_t, c_t)

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

            c_i : torch.Tensor of shape (batch, input_size)
                Tensor containing the next cell state
            """
        # Pass through network
        f_t  = torch.sigmoid(x @ self.W_f  + h_t @ self.U_f  + self.b_f )
        i_t  = torch.sigmoid(x @ self.W_i  + h_t @ self.U_i  + self.b_i )
        o_t  = torch.sigmoid(x @ self.W_o  + h_t @ self.U_o  + self.b_o )
        c_t_ = torch.tanh   (x @ self.W_c_ + h_t @ self.U_c_ + self.b_c_)
        c_t  = f_t * c_t + i_t * c_t_
        h_t  = o_t * torch.tanh(c_t)

        # Return result
        return h_t, c_t

    ########################################################################
    #                        Weight initialisation                         #
    ########################################################################

    def init_weights(self):
        """Initialise weights"""
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    ########################################################################
    #                     Hidden state initialisation                      #
    ########################################################################

    def initHidden(self, x):
        """Initialise hidden layer"""
        return torch.zeros(x.shape[0], self.hidden_size).to(x.device),\
               torch.zeros(x.shape[0], self.hidden_size).to(x.device)

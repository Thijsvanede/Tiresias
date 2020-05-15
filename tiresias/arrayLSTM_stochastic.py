# Import pytorch library
import torch
import torch.nn.functional as F
# Import standard arrayLSTM implementation
from arrayLSTM import ArrayLSTM

class StochasticArrayLSTM(ArrayLSTM):

    """Implementation of ArrayLSTM with Stochastic Output Pooling

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
        # Compute softmax s
        probability = F.softmax(outputs, dim=0)
        # Get maximum probability
        probability = probability.max(dim=0).indices

        # Initialise output
        output = torch.zeros(probability.shape, device=states.device)
        state  = torch.zeros(probability.shape, device=states.device)
        # Fill output
        for k in range(self.k):
            # Create mask
            mask = probability == k
            # Fill output and state
            output[mask] = outputs[k][mask]
            state [mask] = states [k][mask]

        # Compute hidden and return
        hidden = torch.mul(output, state.tanh())
        # Reshape hidden
        hidden = hidden.unsqueeze_(0)
        # Return hidden
        return hidden

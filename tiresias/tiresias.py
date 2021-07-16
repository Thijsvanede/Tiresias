from arrayLSTM            import      LSTM
from arrayLSTM            import ArrayLSTM
from arrayLSTM.extensions import  AttentionArrayLSTM
from arrayLSTM.extensions import StochasticArrayLSTM
from torchtrain import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        # self.lstm    = ArrayLSTM(input_size, hidden_size, k)
        self.lstm    = StochasticArrayLSTM(input_size, hidden_size, k)
        self.linear  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        """Forward data through the network

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Input of sequences, these will be one-hot encoded to an array of
                shape=(n_samples, seq_len, input_size)

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, size_out)
                Returns a probability distribution of the possible outputs
            """
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

    def predict(self, X, k=1, variable=False, verbose=True):
        """Predict the k most likely output values

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Input of sequences, these will be one-hot encoded to an array of
                shape=(n_samples, seq_len, input_size)

            k : int, default=1
                Number of output items to generate

            variable : boolean, default=False
                If True, predict inputs of different sequence lengths

            verbose : boolean, default=True
                If True, print output

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, k)
                k most likely outputs

            confidence : torch.Tensor of shape=(n_samples, k)
                Confidence levels for each output
            """
        # Get the predictions
        result = super().predict(X, variable=variable, verbose=verbose)
        # Get the probabilities from the log probabilities
        result = result.exp()
        # Compute k most likely outputs
        confidence, result = result.topk(k)
        # Return result
        return result, confidence


    def predict_online(self, X, y,
        k             = 1,
        epochs        = 10,
        batch_size    = 32,
        learning_rate = 0.0001,
        criterion     = nn.NLLLoss(),
        optimizer     = optim.SGD,
        variable      = False,
        verbose       = True,
        **kwargs):
        """Predict samples in X and update the network only if the prediction
            does not match y

            Parameters
            ----------
            X : torch.Tensor
                Tensor to predict/train with

            y : torch.Tensor
                Target tensor

            k : int, default=1
                Number of output items to generate

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=32
                Default batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for optimizer

            criterion : nn.Loss, default=nn.NLLLoss
                Loss function to use

            optimizer : optim.Optimizer, default=optim.SGD
                Optimizer to use for training

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, k)
                k most likely outputs

            confidence : torch.Tensor of shape=(n_samples, k)
                Confidence levels for each output
            """
        # Initialise output
        result     = list()
        confidence = list()

        # Loop over each batch
        for batch in range(0, len(X), batch_size):
            # Extract batch
            X_ = X[batch:batch+batch_size]
            y_ = y[batch:batch+batch_size]

            # Get prediction
            y_pred_, confidence_ = self.predict(X_, k,
                variable=variable, verbose=False)

            # Add prediction
            result    .append(y_pred_    )
            confidence.append(confidence_)

            # Check if prediction matches output
            match = y_pred_[:, 0] == y_
            for i in range(1, k):
                match |= y_pred_[:, i] == y_

            # Update non-matching output if any
            if match.sum() != match.shape[0]:
                self.fit(X_[~match], y_[~match],
                    epochs        = epochs,
                    batch_size    = batch_size,
                    learning_rate = learning_rate,
                    criterion     = criterion,
                    optimizer     = optimizer,
                    variable      = variable,
                    verbose       = verbose,
                    **kwargs)

        # Concatenate outputs and return
        result     = torch.cat(result    , dim=0)
        confidence = torch.cat(confidence, dim=0)
        # Return result
        return result, confidence

    ########################################################################
    #                           Save/load model                            #
    ########################################################################

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        # Save to output file
        torch.save(self.state_dict(), outfile)

    @classmethod
    def load(cls, infile, device=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.
            """
        # Load state dictionary
        state_dict = torch.load(infile, map_location=device)

        # Get input variables from state_dict
        input_size  = state_dict.get('lstm.i2h.weight').shape[1]
        hidden_size = state_dict.get('lstm.h2h.weight').shape[1]
        output_size = input_size
        k           = state_dict.get('lstm.i2h.weight').shape[0] // (4*hidden_size)

        # Create ContextBuilder
        result = cls(
            input_size  = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            k           = k,
        )

        # Cast to device if necessary
        if device is not None: result = result.to(device)

        # Set trained parameters
        result.load_state_dict(state_dict)

        # Return result
        return result

import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from variable_data_loader import VariableDataLoader

class Module(nn.Module):
    """Extention of nn.Module that adds fit and predict methods
        Can be used for automatic training.

        Attributes
        ----------
        progress : Progress()
            Used to track progress of fit and predict methods
    """

    def __init__(self, *args, **kwargs):
        # Initialise super
        super().__init__(*args, **kwargs)
        # Keep progress
        self.progress = Progress()

    def fit(self, X, y,
            epochs        = 10,
            batch_size    = 32,
            learning_rate = 0.01,
            criterion     = nn.NLLLoss,
            optimizer     = optim.SGD,
            variable      = False,
            verbose       = True,
            **kwargs):
        """Train the module with given parameters

            Parameters
            ----------
            X : torch.Tensor
                Tensor to train with

            y : torch.Tensor
                Target tensor

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

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : self
                Returns self
            """
        ################################################################
        #                Initialise training parameters                #
        ################################################################
        # Set optimiser
        optimizer = optimizer(
            params = self.parameters(),
            lr     = learning_rate
        )
        # Set criterion
        criterion = criterion()
        # Initialise progress
        self.progress.reset(len(X), epochs)

        ################################################################
        #                         Prepare data                         #
        ################################################################

        # If the input length can be variable
        if variable:
            # Set device automatically
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Load data as variable length dataset
            data = VariableDataLoader(X, y, batch_size=batch_size, shuffle=True)

        # In the normal case
        else:
            # Get device
            device = X.device
            # Load data
            data = DataLoader(
                TensorDataset(X, y),
                batch_size = batch_size,
                shuffle    = True
            )

        ################################################################
        #                       Perform training                       #
        ################################################################

        # Loop over each epoch
        for epoch in range(1, epochs+1):
            # Loop over entire dataset
            for X_, y_ in data:
                # Forward pass
                # Get new input batch
                X_ = X_.clone().detach().to(device)
                # Run through module
                y_pred = self(X_)
                # Compute loss
                loss = criterion(y_pred, y_)

                # Backward pass
                # Clear optimizer
                optimizer.zero_grad()
                # Propagate loss
                loss.backward()
                # Perform optimizer step
                optimizer.step()

                # Update progress
                if verbose: self.progress.update(loss, X_.shape[0])
            # New line for each epoch
            if verbose: self.progress.update_epoch()

        ################################################################
        #                         Returns self                         #
        ################################################################

        # Return self
        return self


    def predict(self, X, variable=False, **kwargs):
        """Makes prediction based on input data X.
            Default implementation just uses the module forward(X) method,
            often the predict method will be overwritten to fit the specific
            needs of the module.

            Parameters
            ----------
            X : torch.Tensor
                Tensor from which to make prediction

            variable : boolean, default=False
                If True, accept inputs of variable length

            Returns
            -------
            result : torch.Tensor
                Resulting prediction
            """
        # If we expect variable input
        if variable:
            # Initialise result
            result  = list()
            indices = list()

            # Load data
            data = VariableDataLoader(X, torch.zeros(len(X)),
                index=True,
                batch_size=len(X),
                shuffle=False
            )

            # Loop over data
            for X_, y_, i in data:
                # Perform prediction and append
                result .append(self(X_))
                # Store index
                indices.append(i)

            # Concatenate inputs
            result  = torch.cat(result)
            indices = torch.cat(indices)

            # Return originally indexed result
            return result[indices]

        # If input is not variable
        else:
            return self(X)


    def fit_predict(self, X, y,
            epochs        = 10,
            batch_size    = 32,
            learning_rate = 0.01,
            criterion     = nn.NLLLoss,
            optimizer     = optim.SGD,
            verbose       = True,
            **kwargs):
        """Train the module with given parameters

            Parameters
            ----------
            X : torch.Tensor
                Tensor to train with

            y : torch.Tensor
                Target tensor

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

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : torch.Tensor
                Resulting prediction
            """
        return self.fit(X, y,
                        epochs,
                        batch_size,
                        learning_rate,
                        criterion,
                        optimizer,
                        verbose,
                        **kwargs
            ).predict(X, **kwargs)



################################################################################
#                            Keep track of progress                            #
################################################################################

class Progress(object):

    def __init__(self, size=-1, epochs=-1):
        """Track progress of NN training"""
        # Reset progress
        self.reset(size, epochs)

    def reset(self, size, epochs):
        """Reset progress for a given training size and epochs

            Parameters
            ----------
            size : int
                Number of items in training data

            epochs : int
                Number of epochs to train with
            """
        # Set variables
        self.size     = size
        self.epochs   = epochs

        # Set progress
        self.epoch = 0
        self.start = time.time()
        self.last  = time.time()

        # Track loss values
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

        # Return self
        return self

    def update(self, val, n=1):
        """Update self and print progress

            Parameters
            ----------
            val : float
                Loss value to update with

            n : int
                Number of items to update with

            epoch : boolean, default=False
                Whether to update to a new epoch
            """
        # Perform check on initialisation
        if self.size < 0 or self.epochs < 0:
            raise ValueError("Progress has not yet been initialised, call "
                             "reset() first.")

        # Update loss
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

        if (time.time() - self.last) >= 0.05:
            self.last = time.time()
            print(self, end='\r')

        # Return self
        return self

    def update_epoch(self):
        """Update to new epoch"""
        # Extract current epoch and sizes
        epoch  = self.epoch

        # Print newline
        print(self)

        # Reset self
        self.reset(self.size, self.epochs)
        # Update epochs
        self.epoch  = epoch+1

        # Return self
        return self

    def time_since(self, now=None):
        """Get time since start

            Parameters
            ----------
            now : datetime, default=None
                If given, compute difference between start and given time

            Returns
            -------
            time : timedelta
                Difference in time between start and now
            """
        # Get time difference
        return datetime.timedelta(seconds=(now or time.time()) - self.start)


    def __str__(self):
        """Get string representation of progress"""
        # Compute timedelta
        time_diff = self.time_since().total_seconds()
        hours     = int(time_diff // 3600)
        time_diff = time_diff - (hours*3600)
        minutes   = int(time_diff // 60)
        time_diff = time_diff - (minutes*60)
        seconds   = int(time_diff)
        millis    = int(10*(time_diff - seconds))
        time_diff = "{}:{:02}:{:02}.{}".format(hours, minutes, seconds, millis)

        return "[Epoch {:{width}}/{:{width}}] average loss = {:.4f} "\
               "{}{} ({:6.2f}%) runtime {}".format(
               # Epoch and loss information
               self.epoch+1, self.epochs, self.avg,
               # Progress in percentage
               "#" * round(40*    (min(self.size, self.count))/self.size  ),
               "." * round(40* (1-(min(self.size, self.count))/self.size) ),
               100*(min(self.size, self.count))/self.size,
               # Progress in time
               time_diff,
               width=len(str(self.epochs)))

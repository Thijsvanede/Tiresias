import torch
from torch.utils.data import DataLoader, TensorDataset

class VariableDataLoader(object):

    def __init__(self, X, y, batch_size=1, shuffle=True):
        """Load data from variable length inputs

        Parameters
        ----------
        X : iterable of shape=(n_samples,)
            Input sequences
            Each item in iterable should be a sequence (of variable length)

        y : iterable of shape=(n_samples,)
            Labels corresponding to X

        batch_size : int, default=1
            Size of each batch to output

        shuffle : boolean, default=True
            If True, shuffle the data randomly, each yielded batch contains
            only input items of the same length
        """
        # Get inputs by length
        self.lengths = dict()
        # Loop over inputs
        for X_, y_ in zip(X, y):
            X_length, y_length = self.lengths.get(len(X_), (list(), list()))
            X_length.append(X_)
            y_length.append(y_)
            self.lengths[len(X_)] = (X_length, y_length)

        # Transform to tensors
        for k, v in self.lengths.items():
            self.lengths[k] = (torch.as_tensor(v[0]), torch.as_tensor(v[1]))

        # Set batch_size
        self.batch_size = batch_size
        # Set shuffle
        self.shuffle = shuffle
        # Reset
        self.reset()

        # Get keys
        self.keys = set(self.data.keys())

    def reset(self):
        """Reset the VariableDataLoader"""
        # Reset done
        self.done = set()
        # Reset DataLoaders
        self.data = { k: iter(DataLoader(
            TensorDataset(v[0], v[1]),
            batch_size = self.batch_size,
            shuffle    = self.shuffle))
            for k, v in self.lengths.items()
        }

    def __iter__(self):
        """Returns iterable of VariableDataLoader"""
        # Reset
        self.reset()
        # Return self
        return self

    def __next__(self):
        """Get next item of VariableDataLoader"""
        # Check if we finished the iteration
        if self.done == self.keys:
            self.reset()
            # Stop iterating
            raise StopIteration

        # Select key
        if self.shuffle:
            key = next(iter(self.keys - self.done))
        else:
            key = next(sorted(self.keys - self.done))

        # Yield next item in batch
        try:
            item = next(self.data.get(key))
        except StopIteration:
            # Add key
            self.done.add(key)
            # Get item iteratively
            item = next(self)

        # Return next item
        return item

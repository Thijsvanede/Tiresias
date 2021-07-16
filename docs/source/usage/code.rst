Code
====
To use Tiresias into your own project, you can use it as a standalone module.
Here we show some simple examples on how to use the Tiresias package in your own python code.
For a complete documentation we refer to the :ref:`Reference` guide.

Import
^^^^^^
To import components from Tiresias simply use the following format

.. code:: python

  from tiresias          import <Object>
  from tiresias.<module> import <Object>

For example, the following code imports the Tiresias neural network as found in the :ref:`Reference`.

.. code:: python

  # Imports
  from tiresias import Tiresias

Working example
^^^^^^^^^^^^^^^

In this example, we load data from either a ``.csv`` or ``.txt`` file and use that data to train and predict with Tiresias.

.. code:: python

  # import Tiresias and Preprocessor
  from tiresias              import Tiresias
  from tiresias.preprocessor import Preprocessor

  ##############################################################################
  #                                 Load data                                  #
  ##############################################################################

  # Create preprocessor for loading data
  preprocessor = Preprocessor(
      length  = 20,           # Extract sequences of 20 items
      timeout = float('inf'), # Do not include a maximum allowed time between events
  )

  # Load data from csv file
  y, X, label, mapping = preprocessor.csv("<path/to/file.csv>")
  # Load data from txt file
  y, X, label, mapping = preprocessor.txt("<path/to/file.txt>")

  ##############################################################################
  #                                  Tiresias                                  #
  ##############################################################################

  # Create Tiresias object
  tiresias = Tiresias(
      input_size  = 300, # Number of different events to expect
      hidden_size = 128, # Hidden dimension, we suggest 128
      output_size = 300, # Number of different events to expect
      k           = 4,   # Number of parallel LSTMs for ArrayLSTM
  )

  # Optionally cast data and Tiresias to cuda, if available
  tiresias = tiresias.to("cuda")
  X        = X       .to("cuda")
  y        = y       .to("cuda")

  # Train tiresias
  tiresias.fit(
      X          = X,
      y          = y,
      epochs     = 10,
      batch_size = 128,
  )

  # Predict using tiresias
  y_pred, confidence = tiresias.predict_online(
      X = X,
      y = y,
      k = 3,
  )

Modifying Tiresias
^^^^^^^^^^^^^^^^^^

Tiresias itself works with an LSTM as implemented by ArrayLSTM from the `array-lstm` package.
Suppose that we want to use a regular LSTM instead, we can simply create a new class that extends Tiresias and overwrite the ``__init__`` method to replace the ArrayLSTM with a regular LSTM.

.. code:: python

  # Imports
  import torch.nn as nn
  from tiresias import Tiresias

  # Create a new class of Tiresias to overwrite the original
  class TiresiasLSTM(Tiresias):

    # We overwrite the __init__ method
    def __init__(self, input_size, hidden_size, output_size, k):
          # Initialise super
          super().__init__(input_size, hidden_size, output_size, k)

          # Replace the lstm layer with a regular LSTM
          self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

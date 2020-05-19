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

In this example, we import all different LSTM implementations and use it to predict the next item in a sequence.
First we import the necessary torch modules and different LSTMs that we want to use.

.. code:: python

  # import Tiresias and PreprocessLoader
  from tiresias import Tiresias
  from tiresias.processing import PreprocessLoader

  ##############################################################################
  #                                 Load data                                  #
  ##############################################################################
  # Create loader for preprocessed data
  loader = PreprocessLoader()
  # Load data
  data, encodings = loader.load(
      <infile>,
      dim_in      = 20,
      dim_out     = 1,
      train_ratio = 0.5,
      key         = lambda x: (x.get(<groupby_key>),),
      extract     = [<event_key>],
      random      = False
  )

  # Get short handles
  X_train = data.get('threat_name').get('train').get('X').to(device)
  y_train = data.get('threat_name').get('train').get('y').to(device).reshape(-1)
  X_test  = data.get('threat_name').get('test' ).get('X').to(device)
  y_test  = data.get('threat_name').get('test' ).get('y').to(device).reshape(-1)

  ##############################################################################
  #                                  Tiresias                                  #
  ##############################################################################
  tiresias = Tiresias(args.input, args.hidden, args.input, args.k).to(device)
  # Train tiresias
  tiresias.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
  # Predict using tiresias
  y_pred, confidence = tiresias.predict_online(X_test, y_test, k=args.top)

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

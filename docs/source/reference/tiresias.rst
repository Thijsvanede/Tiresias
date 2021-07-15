.. _Tiresias:

Tiresias
========

The Tiresias class uses the `torch-train`_ library for training and prediction.
This class implements the neural network as described in the paper `Tiresias: Predicting Security Events Through Deep Learning`_.

.. _`Tiresias: Predicting Security Events Through Deep Learning`: https://doi.org/10.1145/3243734.3243811
.. _`torch-train`: https://github.com/Thijsvanede/torch-train


.. autoclass:: tiresias.Tiresias

Initialization
^^^^^^^^^^^^^^

.. automethod:: tiresias.Tiresias.__init__

Forward
^^^^^^^

As Tiresias is a Neural Network, it implements the :py:meth:`forward` method which passes input through the entire network.

.. automethod:: tiresias.Tiresias.forward

Predict
^^^^^^^

The regular network gives a probability distribution over all possible output values.
However, Tiresias outputs the `k` most likely outputs, therefore it overwrites the :py:meth:`predict` method of the :py:class:`Module` class from `torch-train`_.

.. automethod:: tiresias.Tiresias.predict

In addition to regular prediction, Tiresias introduces `online prediction`.
In this implementation, the network predicts outputs for given inputs and compares them to what actually occurred.
If the prediction does not match the actual output event, we update the neural network before predicting the next events.
This is done using the method :py:meth:`predict_online`.

.. automethod:: tiresias.Tiresias.predict_online

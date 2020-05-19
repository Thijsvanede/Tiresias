Installation
============
The most straigtforward way of installing Tiresias is via pip

.. code::

  pip install tiresias

If you wish to stay up to date with the latest development version, you can instead download the `source code`_.
In this case, make sure that you have all the required `dependencies`_ installed.

.. _source code: https://github.com/Thijsvanede/Tiresias

.. _dependencies:

Dependencies
^^^^^^^^^^^^
Tiresias requires the following python packages to be installed:

- array-lstm: https://github.com/Thijsvanede/ArrayLSTM
- numpy: https://numpy.org/
- scikit-learn: https://scikit-learn.org/
- pytorch: https://pytorch.org/

All dependencies should be automatically downloaded if you install Tiresias via pip. However, should you want to install these libraries manually, you can install the dependencies using the requirements.txt file

.. code::

  pip install -r requirements.txt

Or you can install these libraries yourself

.. code::

  pip install -U array-lstm numpy scikit-learn torch

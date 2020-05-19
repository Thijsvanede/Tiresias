.. _Utils:

Utils
=====

We provide an additional utility that properly formats the argparse of main.

.. autoclass:: utils.TextHelpFormatter

Usage
^^^^^
In order to use the TextHelpFormatter in your argument parser,

.. code:: python

  # Import argparse
  import argparse
  # Import TextHelpFormatter
  from tiresias.utils import TextHelpFormatter

  # Create argument parser
  parser = argparse.ArgumentParser(
      prog        = "<Your program name>",
      description = "<Your program description>",
      formatter_class=TextHelpFormatter
  )

  # Add arguments...

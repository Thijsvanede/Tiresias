Command line tool
=================
When Tiresias is installed, it can be used from the command line.
The :code:`__main__.py` file in the :code:`tiresias` module implements this command line tool.
The command line tool provides a quick and easy interface to predict sequences from :code:`.csv` files.
The full command line usage is given in its :code:`help` page:

.. Note::

  Note that when handling very large inputs, Tiresias is very slow.
  In order to more quickly test on smaller inputs we provide the ``--max`` flag, which specifies the maximum amount of samples to read from the input file.
  E.g., to use only the first 100k samples, one may invoke Tiresias using ``--max 1e5`` flag.

.. code:: text

  usage: tiresias.py [-h] [-f FIELD] [-l LENGTH] [-m MAX] [--hidden HIDDEN] [-i INPUT] [-k K] [-t TOP] [-b BATCH_SIZE]
                   [-d DEVICE] [-e EPOCHS] [-r] [--ratio RATIO]
                   file

  Tiresias: Predicting Security Events Through Deep Learning

  optional arguments:
  -h, --help                   show this help message and exit

  Input parameters:
  file                         file to read as input
  -f, --field      FIELD       FIELD to extract from input FILE           (default = threat_name)
  -l, --length     LENGTH      length of input sequence                   (default =          20)
  -m, --max        MAX         maximum number of items to read from input (default =         inf)

  Tiresias parameters:
  --hidden         HIDDEN      hidden dimension                           (default =         128)
  -i, --input      INPUT       input  dimension                           (default =         300)
  -k, --k          K           number of concurrent memory cells          (default =           4)
  -t, --top        TOP         accept any of the TOP predictions          (default =           1)

  Training parameters:
  -b, --batch-size BATCH_SIZE  batch size                                 (default =         128)
  -d, --device     DEVICE      train using given device (cpu|cuda|auto)   (default =        auto)
  -e, --epochs     EPOCHS      number of epochs to train with             (default =          10)
  -r, --random                 train with random selection
  --ratio          RATIO       proportion of data to use for training     (default =         0.5)

Examples
^^^^^^^^
Use first half of ``<data.csv>`` to train Tiresias and use second half of ``<data.csv>`` to predict and test the prediction.

.. code::

  python3 -m tiresias <data.csv> --ratio 0.5

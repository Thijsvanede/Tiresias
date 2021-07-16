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

  usage: tiresias.py [-h] [--csv CSV] [--txt TXT] [--length LENGTH] [--timeout TIMEOUT] [--hidden HIDDEN] [-i INPUT] [-k K] [-o] [-t TOP] [--save SAVE] [--load LOAD] [-b BATCH_SIZE] [-d DEVICE] [-e EPOCHS]
                   {train,predict}

  Tiresias: Predicting Security Events Through Deep Learning

  positional arguments:
  {train,predict}              mode in which to run Tiresias

  optional arguments:
  -h, --help                   show this help message and exit

  Input parameters:
  --csv       CSV              CSV events file to process
  --txt       TXT              TXT events file to process
  --length    LENGTH           sequence LENGTH                          (default =   20)
  --timeout   TIMEOUT          sequence TIMEOUT (seconds)               (default =  inf)

  Tiresias parameters:
  --hidden    HIDDEN           hidden dimension                         (default =  128)
  -i, --input INPUT            input  dimension                         (default =  300)
  -k, --k     K                number of concurrent memory cells        (default =    4)
  -o, --online                 use online training while predicting
  -t, --top   TOP              accept any of the TOP predictions        (default =    1)
  --save      SAVE             save Tiresias to   specified file
  --load      LOAD             load Tiresias from specified file

  Training parameters:
  -b, --batch-size BATCH_SIZE  batch size                               (default =  128)
  -d, --device DEVICE          train using given device (cpu|cuda|auto) (default = auto)
  -e, --epochs EPOCHS          number of epochs to train with           (default =   10)

Examples
^^^^^^^^
Use first half of ``<data.csv>`` to train Tiresias and use second half of ``<data.csv>`` to predict and test the prediction.

.. code::

  python3 -m tiresias train   --csv <data_train.csv> --save tiresias.save # Training
  python3 -m tiresias predict --csv <data_test.csv>  --load tiresias.save # Predicting

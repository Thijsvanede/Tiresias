import argparse
import torch
import warnings
from argformat import StructuredFormatter

from .preprocessor  import Preprocessor
from .tiresias      import Tiresias

if __name__ == "__main__":
########################################################################
#                           Parse arguments                            #
########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog        = "tiresias.py",
        description = "Tiresias: Predicting Security Events Through Deep Learning",
        formatter_class=StructuredFormatter
    )

    # Add Tiresias mode arguments, run in different modes
    parser.add_argument('mode', help="mode in which to run Tiresias", choices=(
        'train',
        'predict',
    ))

    # Add input arguments
    group_input = parser.add_argument_group("Input parameters")
    group_input.add_argument('--csv'      , help="CSV events file to process")
    group_input.add_argument('--txt'      , help="TXT events file to process")
    group_input.add_argument('--length'   , type=int  , default=20          , help="sequence LENGTH           ")
    group_input.add_argument('--timeout'  , type=float, default=float('inf'), help="sequence TIMEOUT (seconds)")

    # Tiresias
    group_tiresias = parser.add_argument_group("Tiresias parameters")
    group_tiresias.add_argument(      '--hidden', type=int, default=128, help='hidden dimension')
    group_tiresias.add_argument('-i', '--input' , type=int, default=300, help='input  dimension')
    group_tiresias.add_argument('-k', '--k'     , type=int, default=4  , help='number of concurrent memory cells')
    group_tiresias.add_argument('-o', '--online', action='store_true'  , help='use online training while predicting')
    group_tiresias.add_argument('-t', '--top'   , type=int, default=1  , help='accept any of the TOP predictions')
    group_tiresias.add_argument('--save', help="save Tiresias to   specified file")
    group_tiresias.add_argument('--load', help="load Tiresias from specified file")

    # Training
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('-b', '--batch-size', type=int, default=128,   help="batch size")
    group_training.add_argument('-d', '--device'    , default='auto'     ,     help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10,    help="number of epochs to train with")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Set device
    if args.device is None or args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = args.length,
        timeout = args.timeout,
    )

    # Load files
    if args.csv is not None and args.txt is not None:
        # Raise an error if both csv and txt are specified
        raise ValueError("Please specify EITHER --csv OR --txt.")
    if args.csv:
        # Load csv file
        X, y, label, mapping = preprocessor.csv(args.csv)
    elif args.txt:
        # Load txt file
        X, y, label, mapping = preprocessor.txt(args.txt)

    X = X.to(args.device)
    y = y.to(args.device)

    ########################################################################
    #                               Tiresias                               #
    ########################################################################

    # Create instance of Tiresias
    tiresias = Tiresias(
        input_size  = args.input,
        hidden_size = args.hidden,
        output_size = args.input,
        k           = args.k,
    ).to(args.device)

    # Load Tiresias from file, if necessary
    if args.load:
        tiresias = Tiresias.load(args.load).to(args.device)

    # Train Tiresias
    if args.mode == "train":

        # Print warning if training Tiresias without saving it
        if args.save is None:
            warnings.warn("Training Tiresias without saving it to output.")

        # Fit Tiresias with data
        tiresias.fit(
            X          = X,
            y          = y,
            epochs     = args.epochs,
            batch_size = args.batch_size,
        )

        # Save Tiresias to file
        if args.save:
            tiresias.save(args.save)

    # Predict with Tiresias
    if args.mode == "predict":
        if args.online:
            y_pred, confidence = tiresias.predict_online(X, y, k=args.top)
        else:
            y_pred, confidence = tiresias.predict(X, k=args.top)

        ####################################################################
        #                         Show predictions                         #
        ####################################################################

        # Initialise predictions
        y_pred_top = y_pred[:, 0].clone()
        # Compute top TOP predictions
        for top in range(1, args.top):
            print(top, y_pred.shape)
            # Get mask
            mask = y == y_pred[:, top]
            # Set top values
            y_pred_top[mask] = y[mask]

        from sklearn.metrics import classification_report
        print(classification_report(y.cpu(), y_pred_top.cpu(), digits=4))

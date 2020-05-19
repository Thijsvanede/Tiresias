import argparse
import torch
from preprocessing import Preprocessor, PreprocessLoader
from tiresias import Tiresias
from utils import TextHelpFormatter

if __name__ == "__main__":
########################################################################
#                           Parse arguments                            #
########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog        = "tiresias.py",
        description = "Tiresias: Predicting Security Events Through Deep Learning",
        formatter_class=TextHelpFormatter
    )

    # Add arguments
    group_input = parser.add_argument_group("Input parameters")
    group_input.add_argument('file', help='file to read as input')
    group_input.add_argument('-m', '--max'   , type=float, default=float('inf'), help='maximum number of items to read from input')
    group_input.add_argument('-l', '--length', type=int  , default=20          , help="length of input sequence")

    # Tiresias
    group_tiresias = parser.add_argument_group("Tiresias parameters")
    group_tiresias.add_argument('-i', '--input' , type=int, default=300, help='input  dimension')
    group_tiresias.add_argument(      '--hidden', type=int, default=128, help='hidden dimension')
    group_tiresias.add_argument('-k', '--k'     , type=int, default=4  , help='number of concurrent memory cells')
    group_tiresias.add_argument('-t', '--top'   , type=int, default=1  , help='accept any of the TOP predictions')

    # Training
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('-b', '--batch-size', type=int, default=128, help="batch size")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10,  help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true',   help="train with random selection")
    group_training.add_argument('-d', '--device'    , default='auto'     ,   help="train using given device (cpu|cuda|auto)")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Set device
    if args.device is None or args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create loader for preprocessed data
    loader = PreprocessLoader()
    # Load data
    data, encodings = loader.load(args.file, args.length, 1,
                                  args.max, train_ratio=0.5, random=args.random)

    # Get short handles
    X_train = data.get('threat_name').get('train').get('X').to(device)
    y_train = data.get('threat_name').get('train').get('y').to(device).reshape(-1)
    X_test  = data.get('threat_name').get('test' ).get('X').to(device)
    y_test  = data.get('threat_name').get('test' ).get('y').to(device).reshape(-1)

    ########################################################################
    #                               Tiresias                               #
    ########################################################################
    tiresias = Tiresias(args.input, args.hidden, args.input, args.k).to(device)
    # Train tiresias
    tiresias.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    # Predict using tiresias
    y_pred, confidence = tiresias.predict(X_test, k=args.top)

    ########################################################################
    #                           Show predictions                           #
    ########################################################################
    # Initialise predictions
    y_pred_top = y_pred[:, 0].clone()
    # Compute top TOP predictions
    for top in range(1, args.top):
        print(top, y_pred.shape)
        # Get mask
        mask = y_test == y_pred[:, top]
        # Set top values
        y_pred_top[mask] = y_test[mask]

    from sklearn.metrics import classification_report
    print(classification_report(y_test.cpu(), y_pred_top.cpu(), digits=4))

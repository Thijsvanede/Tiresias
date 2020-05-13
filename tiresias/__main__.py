import argparse
from preprocessing import Preprocessor
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
    group_input.add_argument('-m', '--max', type=float, default=float('inf'), help='maximum number of items to read from input')

    group_tiresias = parser.add_argument_group("Tiresias parameters")
    group_tiresias.add_argument('-i', '--input' , default=300, help='input  dimension')
    group_tiresias.add_argument('-l', '--hidden', default=128, help='hidden dimension')
    group_tiresias.add_argument('-k', '--k'     , default=4  , help='number of concurrent memory cells')

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Load data                               #
    ########################################################################
    # Initialse preprocessor
    preprocessor = Preprocessor()
    # Load data
    D = preprocessor.load(args.file, max=args.max, min_seq_length=20)

    # TODO proper split
    X_train = [x[:-1] for x in D.values()]
    y_train = [x[ -1] for x in D.values()]
    X_test  = [x[:-1] for x in D.values()]
    import torch
    y_test  = torch.as_tensor([x[ -1] for x in D.values()])

    ########################################################################
    #                               Tiresias                               #
    ########################################################################
    tiresias = Tiresias(args.input, args.hidden, args.input, args.k)
    # Train tiresias
    tiresias.fit(X_train, y_train, batch_size=128, variable=True)
    # Predict using tiresias
    y_pred = tiresias.predict(X_test, variable=True)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, digits=4))

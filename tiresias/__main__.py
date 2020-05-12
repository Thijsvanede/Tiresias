import argparse
from preprocessing import Preprocessor
from utils import TextHelpFormatter

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog        = "tiresias.py",
        description = "Tiresias: Predicting Security Events Through Deep Learning",
        formatter_class=TextHelpFormatter
    )

    # Add arguments
    parser.add_argument('file', help='file to read as input')
    parser.add_argument('-m', '--max', type=float, default=float('inf'), help='maximum number of items to read from input')

    # Parse arguments
    args = parser.parse_args()

    # Initialse preprocessor
    preprocessor = Preprocessor()
    # Load data
    D = preprocessor.load(args.file, max=args.max)

    for k, v in D.items():
        print(k, v)

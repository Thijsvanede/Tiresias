# import Tiresias and Preprocessor
from tiresias              import Tiresias
from tiresias.preprocessor import Preprocessor

##############################################################################
#                                 Load data                                  #
##############################################################################

# Create preprocessor for loading data
preprocessor = Preprocessor(
    length  = 20,           # Extract sequences of 20 items
    timeout = float('inf'), # Do not include a maximum allowed time between events
)

# Load data from csv file
X, y, label, mapping = preprocessor.csv("<path/to/file.csv>")
# Load data from txt file
X, y, label, mapping = preprocessor.txt("<path/to/file.txt>")

##############################################################################
#                                  Tiresias                                  #
##############################################################################

# Create Tiresias object
tiresias = Tiresias(
    input_size  = 300, # Number of different events to expect
    hidden_size = 128, # Hidden dimension, we suggest 128
    output_size = 300, # Number of different events to expect
    k           = 4,   # Number of parallel LSTMs for ArrayLSTM
)

# Optionally cast data and Tiresias to cuda, if available
tiresias = tiresias.to("cuda")
X        = X       .to("cuda")
y        = y       .to("cuda")

# Train tiresias
tiresias.fit(
    X          = X,
    y          = y,
    epochs     = 10,
    batch_size = 128,
)

# Predict using tiresias
y_pred, confidence = tiresias.predict_online(
    X = X,
    y = y,
    k = 3,
)

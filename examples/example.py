# Import pytorch
import torch

# Import Tiresias and Preprocessor
from tiresias              import Tiresias
from tiresias.preprocessor import Preprocessor

###############################################################################
#                                  Load data                                  #
###############################################################################

# Create preprocessor for loading data
preprocessor = Preprocessor(
    length  = 20,           # Extract sequences of 20 items
    timeout = float('inf'), # Do not include a maximum allowed time between events
)

# Load data from csv file
X, y, label, mapping = preprocessor.csv("/home/thijs/Documents/research/robustness/data/preprocessed_small.csv")
# X, y, label, mapping = preprocessor.csv("<path/to/file.csv>")
# Load data from txt file
# X, y, label, mapping = preprocessor.txt("<path/to/file.txt>")

################################################################################
#                                  Split data                                  #
################################################################################

# Split into train and test sets (20:80) by time - assuming events are ordered chronologically
X_train  = X[:X.shape[0]//5 ]
X_test   = X[ X.shape[0]//5:]
y_train  = y[:y.shape[0]//5 ]
y_test   = y[ y.shape[0]//5:]

################################################################################
#                                   Tiresias                                   #
################################################################################

# Create Tiresias object
tiresias = Tiresias(
    input_size  = 300, # Number of different events to expect
    hidden_size = 128, # Hidden dimension, we suggest 128
    output_size = 300, # Number of different events to expect
    k           = 4,   # Number of parallel LSTMs for ArrayLSTM
)

# Optionally cast data and Tiresias to cuda, if available
if torch.cuda.is_available():
    tiresias = tiresias.to("cuda")
    X_train  = X_train .to("cuda")
    y_train  = y_train .to("cuda")
    X_test   = X_test  .to("cuda")
    y_test   = y_test  .to("cuda")

# Train tiresias
tiresias.fit(
    X          = X_train,
    y          = y_train,
    epochs     = 10,
    batch_size = 128,
)

# Predict using tiresias
# y_pred, confidence = tiresias.predict_online(
#     X = X_test,
#     y = y_test,
#     k = 3,
# )

# Predict using tiresias
y_pred, confidence = tiresias.predict(
    X = X_test,
    k = 3,
)

print(y_pred)
print(y_pred.shape)

################################################################################
#                                  Evaluation                                  #
################################################################################

# Check if correct prediction is in top K
y_pred_ = y_pred[:, 0].detach().clone()

# Loop over all K > 1
for index in range(1, y_pred.shape[1]):
    # Get mask of correct predictions
    mask = y_pred[:, index] == y_test
    # Set correct predictions of mask
    y_pred_[mask] = y_test[mask]

# Set prediction
y_pred = y_pred_


# Get test and prediction as numpy array
y_test = y_test.cpu().numpy()
y_pred = y_pred.cpu().numpy()

# Import classification report
from sklearn.metrics import classification_report

# Print classification report
print(classification_report(
    y_true = y_test,
    y_pred = y_pred,
    digits = 4,
))

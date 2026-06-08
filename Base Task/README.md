
# README.md

# FashionMNIST Classification with PyTorch
Overview
This project trains a custom neural network on the FashionMNIST dataset using PyTorch.
The model uses a branching architecture with skip connections and concatenation, optimized with Adam and evaluated using CrossEntropyLoss.
It saves the best model based on validation loss and generates predictions for the test set.

# Features
Preprocessing with ToTensor and Normalize (scales to [-1, 1]).

Train/validation split (80/20).

Custom neural network:

Flatten layer

Hidden layer

Left branch (with skip connection)

Right branch

Concatenation + output

Training loop with loss tracking.

Model checkpointing (best_model.pkl).

Accuracy evaluation on test set.

Submission file (submission.csv) with predictions.

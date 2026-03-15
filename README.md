# MNIST Image Classification

Classify handwritten digits (0–9) using a small PyTorch neural network trained on the MNIST dataset.

## Goal

Train a 2-layer MLP on MNIST: load data, train for 10 epochs with Adam, and report test accuracy.

## How to run

1. Install dependencies: `torch`, `torchvision`
2. Open `main.ipynb` in Jupyter or a compatible environment
3. Update the data path in the dataset cell if needed (default is a Colab path; use e.g. `./data` locally)
4. Run all cells in order

The notebook downloads MNIST, trains the model, and prints loss and test accuracy per epoch.

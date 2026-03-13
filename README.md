# Image Classification with Neural Network Model – PyTorch

A complete end-to-end implementation of an image classification pipeline for the **MNIST handwritten-digit dataset** using a fully-connected neural network built with **PyTorch**.

---

## 📌 Project Overview

This project walks through every stage of a supervised deep-learning workflow:

| Stage | Description |
|---|---|
| **Data loading** | Downloads the MNIST dataset automatically via `torchvision.datasets.MNIST` and wraps it in `DataLoader` for efficient mini-batch iteration. |
| **Pre-processing** | Normalizes pixel values using the dataset-wide mean (0.1307) and standard deviation (0.3081). |
| **Model definition** | Three-layer fully-connected network (`NeuralNet`) with ReLU activations and a dropout layer for regularisation. |
| **Training** | Optimizes weights with the **Adam** optimizer and **CrossEntropyLoss** over multiple epochs. |
| **Evaluation** | Reports loss and accuracy on the held-out test set after each epoch. |
| **Model saving** | Persists the trained weights to `mnist_model.pth` for later inference. |

---

## 🗂️ Repository Structure

```
.
├── main.py          # Full training and evaluation pipeline
├── data/            # MNIST data downloaded here automatically
├── mnist_model.pth  # Saved model weights (generated after training)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.12
- torchvision ≥ 0.13

Install dependencies:

```bash
pip install torch torchvision
```

### Run the training

```bash
python main.py
```

The script will:
1. Download MNIST to `./data/` on the first run.
2. Train the model for 5 epochs, printing loss and accuracy per epoch.
3. Save the trained model to `mnist_model.pth`.

---

## 🧠 Model Architecture

```
Input  →  Flatten (784)
       →  Linear(784 → 256)  →  ReLU  →  Dropout(0.2)
       →  Linear(256 → 256)  →  ReLU
       →  Linear(256 → 10)
       →  (CrossEntropyLoss applies Softmax internally)
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 5 |
| Hidden layer size | 256 |
| Dropout rate | 0.2 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |

---

## 📊 Expected Results

After 5 epochs on CPU/GPU the model typically achieves **≥ 97 % test accuracy** on the 10 000 MNIST test images.

---

## 📄 License

This project is open-source and available for educational use.

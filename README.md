# Deep Learning Models from Scratch

A clean, modular implementation of Deep Neural Networks entirely from scratch using Python and NumPy. Codebase demonstrates the core mathematical principles behind deep learning, including forward propagation, backpropagation, and gradient descent, without relying on high-level frameworks like **TensorFlow** or **PyTorch**.

## Features

*   **Dynamic Architecture**: Easily configure the number of hidden layers and neurons per layer.
*   **Customizable Activations**: Supports decoupled activation functions.

## Project Structure

*   `nn.py`: Contains the `DeepNN` class, which handles network initialization, forward propagation, and backpropagation calculations based on the chain rule.
*   `utils.py`: Contains activation functions, their backward derivatives, and loss functions (e.g., vectorized binary cross-entropy).
*   `test.ipynb`: A sample Jupyter Notebook demonstrating how to use the framework to classify a non-linear dataset.

## Quick Start

### 1. Requirements

Make sure you have NumPy and Scikit-Learn (for dummy datasets) installed:
```bash
pip install numpy scikit-learn matplotlib
```

#### 2. Run the `test.ipynb` notebook:

```bash
jupyter notebook test.ipynb
```
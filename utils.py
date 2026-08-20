import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_backward(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def binary_crossentropy(y, y_hat):
    epsilon = 1e-15
    y_hat_new = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat_new) + (1 - y) * np.log(1 - y_hat_new))

def categorical_crossentropy(Y, Y_hat):
    epsilon = 1e-15
    Y_hat_clipped = np.clip(Y_hat, epsilon, 1 - epsilon)
    return -np.sum(Y * np.log(Y_hat_clipped)) / Y.shape[1]

def one_hot_encode(Y, num_classes):
    return np.eye(num_classes)[Y].T
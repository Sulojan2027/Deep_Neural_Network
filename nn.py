import numpy as np

class DeepNN:
    """Deep Neural Network classifier
    Params:
        layer_dims: list of integers, number of neurons in each layer
        mid_activation: activation function for hidden layers
        mid_activation_backward: derivative of mid_activation
        out_activation: activation function for output layer
    """
    def __init__(self, layer_dims, mid_activation, mid_activation_backward, out_activation):
        self.parameters = {}
        self.num_layers = len(layer_dims) - 1
        self.mid_activation = mid_activation
        self.mid_activation_backward = mid_activation_backward
        self.out_activation = out_activation

        for i in range(1, self.num_layers + 1):
            self.parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.1
            self.parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    def forward_propagation(self, X):
        caches = {}
        A = X
        caches['A0'] = X
        for i in range(1, self.num_layers + 1):
            W = self.parameters['W' + str(i)]
            b = self.parameters['b' + str(i)]
            Z = np.dot(W, A) + b
            if i == self.num_layers:
                A = self.out_activation(Z)
            else:
                A = self.mid_activation(Z)
            caches['Z' + str(i)] = Z
            caches['A' + str(i)] = A

        return A, caches

    def backward_propagation(self, X, Y, caches):
        grads = {}
        m = X.shape[1]

        # Last layer
        Z = caches['Z' + str(self.num_layers)]
        A = caches['A' + str(self.num_layers)]
        dZ = A - Y
        dW = (1/m) * np.dot(dZ, caches['A' + str(self.num_layers-1)].T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        grads['dW' + str(self.num_layers)] = dW
        grads['db' + str(self.num_layers)] = db

        for i in range(self.num_layers-1, 0, -1):
            Z = caches['Z' + str(i)]
            A_prev = caches['A' + str(i-1)]
            W_next = self.parameters['W' + str(i+1)]
            
            dZ = np.dot(W_next.T, dZ) * self.mid_activation_backward(Z)
            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            grads['dW' + str(i)] = dW
            grads['db' + str(i)] = db
        
        return grads

    def update_parameters(self, grads, learning_rate):
        """Update parameters"""
        for i in range(1, self.num_layers + 1):
            self.parameters['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]

    def predict(self, X):
        """Predict output
        Params:
            X: input data
        Returns:
            predicted output
        """
        A, _ = self._forward_propagation(X)
        return A
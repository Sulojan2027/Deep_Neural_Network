class MyNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.b = 0

    def fit(self, X, y, learning_rate, epochs, loss_threshold):
        self.w1, self.w2, self.b = self._gradient_descent(X['age'], X['affordibility'], y, learning_rate, epochs, loss_threshold)
        print(f"Final weights and bias: w1:{self.w1}, w2: {self.w2}, bias: {self.b}")

    def predict(self,x_test):
        weighted_sum = self.w1 * x_test['age'] + self.w2 * x_test['affordibility'] + self.b
        return sigmoid_numpy(weighted_sum)
    
    def _gradient_descent(self, x1, x2, y, learning_rate, epochs, loss_threshold):
        w1 = w2 = 1
        b = 0
        n = len(x1)
        for i in range(epochs):
            weight = w1 * x1 + w2 * x2 + b
            y_hat = sigmoid_numpy(weight)
            loss  = sigmoid_loss(y, y_hat)

            w1d = np.dot(x1.T, y - y_hat) / n
            w2d = np.dot(x2.T, y - y_hat) / n

            b = np.sum(y - y_hat) / n
            w1 = w1 - learning_rate * w1
            w2 = w2 - learning_rate * w2
            b = b - learning_rate * b
            
            if i % 10 == 0:
                print(f'Epoch: {i}, w1: {w1}, w2: {w2}, bias: {b}, Loss: {loss}')

            if loss < loss_threshold:
                break

        return w1, w2, b

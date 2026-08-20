def sigmoid_numpy(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_loss(y, y_hat):
    epsilon = 1e-15
    y_hat_new = [max(i, epsilon) for i in y_hat]
    y_hat_new = [min(i, 1-epsilon) for i in y_hat_new]
    y_hat_new = np.array(y_hat_new)
    return -np.mean(y * np.log(y_hat_new) + (1 - y) * np.log(1 - y_hat_new))
    # return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))
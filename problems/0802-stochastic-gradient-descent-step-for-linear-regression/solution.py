import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples:int = len(X) 
    
    for i in range(n_iter):
        idx = i % n_samples

        y_hat = X[idx].T @ weights

        L = (y_hat - y[idx])**2

        dL_dy_hat = 2 * (y_hat - y[idx])
        dy_hat_dw = dL_dy_hat * X[idx]

        weights = weights - learning_rate * dy_hat_dw

    return weights

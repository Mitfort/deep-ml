import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    n_samples:int = len(batch_indices)

    X_i = X[batch_indices]

    z = X_i @ weights + bias

    error = z - y[batch_indices]
    L = sum(error**2) / n_samples

    dL_dw = (2 / n_samples) * (X_i.T @ error)
    dL_db = (2 / n_samples) * np.sum(error)

    weights = weights - lr * dL_dw
    bias = bias - lr * dL_db

    return np.concatenate([weights.flatten(),[bias]],axis=0)
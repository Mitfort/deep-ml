import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n,m = X.shape

    W = np.zeros((m,))
    B = 0.0 

    for i in range(max_iter):
        yhat = X @ W + B
        l1 = alpha1 * np.sum(np.abs(W))
        l2 = alpha2 * np.sum(W**2)

        res = yhat - y

        loss = 1 / (2 * n) * np.sum((res)**2) + l1 + l2

        dL_dw = (1/n)* X.T @ res + alpha1 * np.sign(W) + 2 * alpha2 * W

        dL_db = (1/n) * np.sum(res)

        W_new = W - learning_rate * dL_dw
        B_new = B - learning_rate * dL_db

        if np.sum(np.abs(dL_dw)) < tol and abs(B_new - B) < tol:
            W = W_new
            B = B_new
            break

        W = W_new
        B = B_new

    return (W,B)

              

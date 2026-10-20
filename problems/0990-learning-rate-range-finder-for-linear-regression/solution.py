import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    X = np.array(X)
    y = np.array(y)
    w0 = np.array(w0)

    n,m = X.shape
    losses:list[float] = []

    for k in range(n_steps):
        lr:float = 0.0

        if n_steps == 1:
            lr = 10**a
        else:
            lr = 10**(a+(b -a)*k / (n_steps-1))

        preds = X @ w0

        error = preds - y

        grad = (2/n) * X.T @ (error)

        w0 = w0 - lr * grad 

        new_preds = X @ w0 
        error = new_preds - y 

        mse = (1/n) * np.sum(error**2)

        losses.append(mse.item())

    return losses        

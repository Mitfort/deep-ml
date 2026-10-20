import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    
    X = np.array(X)
    y = np.array(y)
    m,n = X.shape
    weights = np.array(weights)

    for epoch in range(n_epochs):

        if method == "batch":
            preds = X @ weights

            error = preds - y 

            mse = (1/(2*m)) * np.sum(error**2)

            grad = (2/m) * X.T @ error

            weights = weights - learning_rate * grad

        elif method == "stochastic":
            for i in range(m):
                pred = X[i] @ weights

                error = pred - y[i]

                mse = error**2

                grad = 2 * X[i] * error

                weights = weights - learning_rate * grad 

        elif method == "mini_batch":
            n_batches = int(np.ceil(m / batch_size))

            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, m)

                x = X[start:end]

                preds = x @ weights

                error = preds - y[start:end]

                mse = (1 / batch_size) * np.sum(error**2)

                grad = 2 / batch_size * x.T @ error 

                weights = weights - learning_rate * grad 


    return weights 



            




        

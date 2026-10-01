import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.array(predictions,dtype=float)
    y_true = np.array(y_true,dtype=float)

    mean_preds:np.ndarray = np.mean(predictions,axis=0)

    squared_bias:float = np.mean((mean_preds - y_true)**2)
    variance:float = np.mean(np.var(predictions,axis=0))
    mse:float = np.mean((predictions - y_true)**2)

    return {
        'bias_squared': squared_bias,
        'variance': variance,
        'mse': mse
    }




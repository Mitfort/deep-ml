import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        return np.corrcoef(X, rowvar=False)
    
    # Compute full joint correlation matrix
    full_corr = np.corrcoef(X, Y, rowvar=False)
    
    # Determine the number of variables (columns) in X
    n_x = X.shape[1] if X.ndim > 1 else 1
    
    # Slice the top-right block (cross-correlation between X and Y)
    return full_corr[:n_x, n_x:]
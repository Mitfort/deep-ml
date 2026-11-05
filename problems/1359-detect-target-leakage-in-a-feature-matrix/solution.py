import numpy as np


def find_leaky_features(X: np.ndarray, y: np.ndarray, threshold: float) -> list:
    """Column indices whose |correlation| with y is >= threshold.

    Args:
        X (np.ndarray): (n, p) feature matrix.
        y (np.ndarray): (n,) target.
        threshold (float): absolute-correlation cutoff.

    Returns:
        list[int]: ascending column indices.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    y_dev = y - y.mean()
    y_ss = np.sqrt(np.sum(y_dev**2))

    leaky = []

    for j in range(X.shape[1]):
        col = X[:,j]
        col_dev = col - col.mean()
        col_ss = np.sqrt(np.sum(col_dev**2))

        if col_ss == 0 or y_ss == 0:
            continue
        
        corr = np.sum(col_dev * y_dev) / (col_ss * y_ss)

        if abs(corr) >= threshold:
            leaky.append(int(j))

    return sorted(leaky)

def is_affine_function_of(col: np.ndarray, y: np.ndarray) -> bool:
    """True when col == a + b * y to within a residual of 1e-9."""
    col = np.asarray(col, dtype=float)
    y = np.asarray(y, dtype=float)

    design = np.column_stack([np.ones(y.shape[0]),y])
    coef,*_ = np.linalg.lstsq(design,col, rcond=None)
    residual = np.sum((design @ coef - col)**2)
    
    return bool(residual < 1e-9)

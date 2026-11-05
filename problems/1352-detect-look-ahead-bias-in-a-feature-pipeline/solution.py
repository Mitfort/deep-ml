import numpy as np


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score using only the trailing `window` values at each point.

    Args:
        x (np.ndarray): 1-D input series.
        window (int): lookback length, inclusive of the current point.

    Returns:
        np.ndarray: same length as x; the first window-1 entries are np.nan.
    """
    res:np.ndarray = np.full(len(x), np.nan, dtype=float)

    def z_score(x:np.ndarray):
        mean = x.mean()
        std = x.std()
        
        if std == 0: std = 1.0

        return (x - mean) / std

    for t in range(window - 1, len(x)):
        current_window = x[t-window+1: t+1]
        res[t] = z_score(current_window)[-1]

    return res

def has_lookahead(feature_fn, x: np.ndarray) -> bool:
    """Return True if feature_fn uses information from the future."""
    
    base = feature_fn(x)

    x_copy = x.copy()

    x_copy[-1] += 1000.0

    new = feature_fn(x_copy)

    return not np.allclose(base[:-1], new[:-1], equal_nan=True)
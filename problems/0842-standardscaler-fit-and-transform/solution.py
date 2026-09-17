import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    
    ROWS,COLS = X_train.shape[0], X_train.shape[1]

    means = np.zeros(COLS)
    stds = np.zeros(COLS)
    
    for col in range(COLS):
        means[col] = sum(X_train[:,col]) / ROWS
        stds[col] = np.sqrt(np.sum((X_train[:,col] - means[col])**2) / ROWS)

    stds[stds==0] = 1.0 

    X_test_scaled = (X_test - means) / stds

    return X_test_scaled


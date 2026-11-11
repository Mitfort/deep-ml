import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    np.random.seed(seed)

    n:int = X.shape[0]

    idxs = np.arange(n)
    np.random.shuffle(idxs)

    train_size = int(n * train_frac)
    val_size = int(n * val_frac)

    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:train_size+val_size]
    test_idxs = idxs[train_size+val_size:]

    x_train = X[train_idxs]
    x_val = X[val_idxs]
    x_test = X[test_idxs]

    y_train = y[train_idxs]
    y_val = y[val_idxs]
    y_test = y[test_idxs]

    x_train = np.column_stack([x_train, np.ones((x_train.shape[0],1))])
    x_test = np.column_stack([x_test, np.ones((x_test.shape[0],1))])

    theta = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

    test_pred = x_test @ theta 

    return (test_pred, train_idxs, val_idxs, test_idxs)




import numpy as np

def cross_val_predict(X, y, k):
    """
    Return out-of-fold predictions for every sample using k-fold
    cross-validation with a linear regression model (with intercept).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n:int = len(y)
    idxs = np.arange(n, dtype=np.int32) 

    base_size = n // k 
    remainder = n % k

    folds = []
    start = 0 

    # Prepare the folds 
    for i in range(k):
        fold_size = base_size + (1 if i < remainder else 0)

        fold = idxs[start:start+fold_size]
        folds.append(fold)

        start += fold_size

    # Run cross-val 
    preds = np.empty(n, dtype=float)

    for i,fold in enumerate(folds):
        x_val = X[fold]
        y_val = y[fold]

        train_idxs = np.concatenate([
            idx for j, idx in enumerate(folds) if j != i
        ])

        x_train = X[train_idxs]
        y_train = y[train_idxs]

        # add bias 
        x_train = np.column_stack([x_train,np.ones((x_train.shape[0]))])
        x_val = np.column_stack([x_val, np.ones((x_val.shape[0]))])

        # OLS 
        theta = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

        # make preds 
        val_preds = x_val @ theta

        preds[fold] = val_preds

    return preds





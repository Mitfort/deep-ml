import numpy as np

def column_transformer(X, transformers):
    """
    Apply per-column-group transformations and horizontally concatenate the results.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        transformers: list of (name, kind, columns) tuples

    Returns:
        2D list of floats with transformed and concatenated columns.
    """
    
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)

    #print(mean)
    res = []

    for name,kind,columns in transformers:
        vals = X[:,columns]

        if kind == "standard":
            vals = (vals - mean[columns]) / std[columns]
            
            res.append(vals)
        
        elif kind == "minmax":
            minvals = xmin[columns]
            maxvals = xmax[columns]

            denom = maxvals - minvals
            denom[denom == 0] = 1.0

            vals = (vals - minvals) / denom

            res.append(vals)
        
        elif kind == "passthrough":
            res.append(vals)

    if not res: 
        return []

    return np.hstack(res).tolist()
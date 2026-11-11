import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    for j in range(X_clean.shape[1]):
        col = X_clean[:,j]
        valid = col[~np.isnan(col)]

        if len(valid) == 0:
            X_clean[:,j] = 0.0
            continue

        mean = np.mean(valid)
        median = np.median(valid)

        q1,q3 = np.percentile(valid, [25,75])
        iqr = q3 - q1 

        if iqr == 0:
            fill_value = median
        else:
            skewness = (q3 + q1 - 2 * median) / iqr 

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_ratio = np.mean(
                (valid < lower) | (valid > upper)
            )

            if abs(skewness) > 0.3 or outlier_ratio > 0.05:
                fill_value = median
            else:
                fill_value = mean 
        
        col[np.isnan(col)] = fill_value
    
    return X_clean
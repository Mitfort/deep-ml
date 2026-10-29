import numpy as np

def ordinal_encode(X, categories):
    """
    Encode categorical features as integer codes using a provided ordering.

    Args:
        X: 2D list/array of shape (n_samples, n_features) with string values.
        categories: list of length n_features; categories[j] is an ordered
                    list of valid categories for column j.

    Returns:
        np.ndarray of shape (n_samples, n_features), dtype int.
        Unknown categories are encoded as -1.
    """

    order_dict = [
        {cat:encoded for encoded, cat in enumerate(cats)}
        for cats in categories
    ]

    X = np.array(X)
    n,m = X.shape

    res = [
        [order_dict[idx].get(cat,-1) for idx,cat in enumerate(X[i])] 
        for i in range(n)
    ]

    return np.array(res)

    

import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    # Generate feature combination indices for degree 0 up to 'degree' inclusive
    all_combs = []
    for d in range(degree + 1):
        all_combs.extend(
            combinations_with_replacement(range(n_features), d)
        )

    # Pre-allocate output matrix
    poly_features = np.zeros((n_samples, len(all_combs)))

    # Compute products across all samples simultaneously
    for idx, comb in enumerate(all_combs):
        if len(comb) == 0:  # Degree 0 (bias term: x^0 = 1)
            poly_features[:, idx] = 1.0
        else:
            poly_features[:, idx] = np.prod(X[:, comb], axis=1)

    # Sort each sample's polynomial features from lowest to highest value
    sorted_features = np.sort(poly_features, axis=1)

    return sorted_features
    

    


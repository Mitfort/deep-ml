import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    
    n,m = X.shape 
    best_gini: float = float('inf')
    best_feature: int = 0
    best_th: float = 0.0
    
    for i in range(m): 
        features = X[:,i]

        thresholds = np.unique(features)

        for th in thresholds:
            left = [y[i] for i,x in enumerate(features) if x <= th]
            right = [y[i] for i,x in enumerate(features) if x > th]

            n_left = len(left)
            n_right = len(right)

            cls_l, count_l = np.unique(left,return_counts=True)
            cls_r, count_r = np.unique(right,return_counts=True)

            probs_l = count_l / n_left
            probs_r = count_r / n_right

            g_left = 1 - np.sum(probs_l**2)
            g_right = 1 - np.sum(probs_r**2)

            G_split = (n_left / n) * g_left + (n_right / n) * g_right

            if G_split < best_gini:
                best_gini = G_split
                best_feature = i
                best_th = th

    return (best_feature, best_th)





    
    
import numpy as np
from collections import Counter

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure X and y are numpy arrays for indexing
    X = np.asarray(X)
    y = np.asarray(y)

    counter = Counter(y)
    classes, counts = zip(*counter.most_common())
    classes = np.array(classes)
    counts = np.array(counts)
    
    train_idxs = []
    test_idxs = []

    # Perform stratified split for each class independently
    for cls in classes:
        cls_idxs = np.where(y == cls)[0]
        np.random.shuffle(cls_idxs)
        
        # Calculate number of test samples for this class
        n_test_cls = int(len(cls_idxs) * test_size)
        
        test_idxs.extend(cls_idxs[:n_test_cls])
        train_idxs.extend(cls_idxs[n_test_cls:])

    # Convert to numpy arrays and shuffle the final index sets
    train_idxs = np.array(train_idxs)
    test_idxs = np.array(test_idxs)
    
    np.random.shuffle(train_idxs)
    np.random.shuffle(test_idxs)

    return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs]
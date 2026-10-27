import numpy as np

def stratified_kfold_indices(y, n_splits):
    """
    Generate train/test indices for stratified K-fold cross-validation.

    Args:
        y: 1D array-like of integer class labels
        n_splits: number of folds

    Returns:
        A list of [train_indices, test_indices] pairs, one per fold.
    """
    y = np.array(y)
    n = len(y)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    classes, counts = np.unique(y, return_counts=True)

    if np.any(counts < n_splits):
        raise ValueError("Each class must have at least n_splits samples")

    grouped_idxs = [np.where(y == c)[0] for c in classes]

    # Podział indeksów każdej klasy na n_splits części
    class_folds = [
        np.array_split(indices, n_splits)
        for indices in grouped_idxs
    ]

    res = []

    for split in range(n_splits):

        # Indeksy testowe z każdej klasy
        test_indices = np.concatenate([
            folds[split] for folds in class_folds
        ])

        # Wszystkie pozostałe indeksy
        train_indices = np.setdiff1d(
            np.arange(n), test_indices
        )

        res.append([sorted(train_indices.tolist()), sorted(test_indices.tolist())])

    return res
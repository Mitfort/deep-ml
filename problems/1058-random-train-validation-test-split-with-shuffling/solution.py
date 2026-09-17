import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n,d = data.shape
    idxs = np.random.default_rng(seed).permutation(n)

    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)

    train = data[idxs[:train_end]]
    val = data[idxs[train_end:validation_end]]
    test = data[idxs[validation_end:]]

    return [train,val,test]
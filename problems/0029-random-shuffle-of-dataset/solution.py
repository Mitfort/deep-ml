import numpy as np

def shuffle_data(X, y, seed=None):
	if seed: np.random.seed(seed)
	
	n:int = X.shape[0]

	idxs = np.arange(n)

	np.random.shuffle(idxs)

	return (X[idxs], y[idxs])
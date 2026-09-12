import numpy as np

def pairwise_cosine_similarity(X):
    n:int = len(X)
    d:int = len(X[0])

    X = np.array(X)
    S = np.zeros((n,n))

    norm = np.linalg.norm(X,axis=1,keepdims=True)

    norm[norm == 0] = 1.0

    X_norm = X / norm

    S = X_norm @ X_norm.T

    return S
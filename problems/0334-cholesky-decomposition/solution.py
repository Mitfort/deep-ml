import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    if not A: return -1 

    A = np.array(A)
    n,m = A.shape 
    
    if n != m: return -1 # Must be a square matrix 

    eigvals = np.linalg.eigvals(A)

    if np.any(eigvals <= 0): return -1 # Must be positive-defined

    # Cholesky 

    L: np.ndarray = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j: 
                sum_k = 0 

                for k in range(i):
                    sum_k += L[i,k]**2
                
                L[i,i] = np.sqrt(A[i,i] - sum_k)
            
            elif i > j: 
                sum_k = 0

                for k in range(j):
                    sum_k += L[i,k] * L[j,k]
                
                L[i,j] = 1 / L[j,j] * (A[i,j] - sum_k)
    
    return L.tolist()




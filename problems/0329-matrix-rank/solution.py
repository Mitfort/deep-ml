import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    A = A.astype(float)
    m, n = A.shape
    rank = 0
    
    for i in range(min(m, n)):
        # Find pivot
        pivot_row = -1
        for j in range(i, m):
            if abs(A[j, i]) > tol:
                pivot_row = j
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]
        
        # Normalize pivot row
        pivot = A[i, i]
        A[i, :] = A[i, :] / pivot
        
        # Eliminate other rows
        for j in range(m):
            if j != i:
                factor = A[j, i]
                A[j, :] = A[j, :] - factor * A[i, :]
        
        rank += 1
    
    return rank
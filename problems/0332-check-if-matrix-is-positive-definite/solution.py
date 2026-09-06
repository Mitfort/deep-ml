import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    
    A = np.array(matrix)

    is_symmetric = np.allclose(A,A.T)

    if is_symmetric:
        eigenvalues = np.linalg.eigvalsh(A)
    else:
        eigenvalues = np.linalg.eigvals(A)

    eigenvalues_sorted = np.sort(eigenvalues)

    is_positive_definite = bool(is_symmetric and np.all(eigenvalues_sorted > 1e-10))

    eigenvalues_rounded = [round(float(val),4) for val in eigenvalues_sorted]   
    
    return {
        'is_positive_definite': is_positive_definite,
        'eigenvalues': eigenvalues_rounded
    }
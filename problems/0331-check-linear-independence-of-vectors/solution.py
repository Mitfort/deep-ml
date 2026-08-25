import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    if not vectors: return True

    matrix = np.array(vectors)

    rows,cols = len(matrix), len(matrix[0])

    rank:int = 0

    for col in range(cols):
        # Find non-zero element in column 
        pivot = None

        for row in range(rank,rows):
            if abs(matrix[row][col] > 1e-10):
                pivot = row
                break
        
        if pivot is None:
            continue
        
        matrix[row],matrix[pivot] = matrix[pivot],matrix[row]

        for row in range(rank + 1, rows):
            factor = matrix[row][col] / matrix[rank][col]

            for j in range(col,cols):
                matrix[row][j] -= factor * matrix[rank][j]

        rank += 1

        if rank == rows:
            break
    
    return rank == len(vectors)

# import numpy as np

# def gaussian_elimination(A, b):
# 	"""
# 	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
# 	:param A: Coefficient matrix
# 	:param b: Right-hand side vector
# 	:return: Solution vector x
# 	"""

# 	ROWS,COLS = A.shape
# 	pivot = A[0,0]
# 	Matrix = np.column_stack([A,b])

# 	for col in range(COLS):
# 		idx = np.argmax(np.abs(Matrix[col:,col])) + col
# 		#Matrix[col,:],Matrix[idx,:] = Matrix[idx,:],Matrix[col,:]
# 		Matrix[[col, idx]] = Matrix[[idx, col]]

# 		pivot = Matrix[col,col]

# 		for row in range(col+1,ROWS):
# 			factor = Matrix[row,col] / pivot
# 			Matrix[row,:] -= factor * Matrix[col,:]
		
# 	X = np.zeros(COLS)
# 	for col in range(COLS-2,0,-1):
# 		known_sum = np.dot(Matrix[col, col + 1:COLS], X[col + 1:COLS])
# 		X[col] = (Matrix[col, -1] - known_sum) / Matrix[col, col]


# 	return X

import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    # Konwersja na float zapobiega błędom typu danych przy działaniach w miejscu (-=)
    A = A.astype(float)
    b = b.astype(float)

    ROWS, COLS = A.shape
    Matrix = np.column_stack([A, b])

    # 1. Eliminacja w przód
    for col in range(COLS):
        # Szukanie pivota od wiersza 'col' w dół (wartość bezwzględna)
        idx = np.argmax(np.abs(Matrix[col:, col])) + col
        
        # Bezpieczna zamiana wierszy w NumPy
        Matrix[[col, idx]] = Matrix[[idx, col]]

        pivot = Matrix[col, col]

        # Eliminacja wierszy poniżej przekątnej
        for row in range(col + 1, ROWS):
            factor = Matrix[row, col] / pivot
            Matrix[row, :] -= factor * Matrix[col, :]

    # 2. Postępowanie odwrotne (od ostatniego wiersza COLS-1 do 0)
    X = np.zeros(COLS)
    for col in range(COLS - 1, -1, -1):
        known_sum = np.dot(Matrix[col, col + 1:COLS], X[col + 1:COLS])
        X[col] = (Matrix[col, -1] - known_sum) / Matrix[col, col]

    return X
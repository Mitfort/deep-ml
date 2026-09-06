import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""

	A = np.array(A)
	ROWS,COLS = A.shape

	Q:np.ndarray = np.zeros((ROWS,COLS))
	R:np.ndarray = np.zeros((COLS,COLS))

	for j in range(COLS):
		v = A[:,j]

		for i in range(j):
			R[i,j] = np.dot(Q[:,i], A[:,j])
			v = v - R[i,j] * Q[:,i]	

		R[j,j] = np.linalg.norm(v)

		Q[:,j] = v / R[j,j]

	return (Q,R) 
	
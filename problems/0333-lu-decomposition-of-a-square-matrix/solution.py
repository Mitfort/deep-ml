import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	
	# Doolittle method 
	A = np.array(A)
	n = A.shape[0]

	L = np.eye(n)
	U = np.zeros((n,n))

	for i in range(n):
		
		for j in range(i,n):
			U[i,j] = A[i,j] - np.dot(L[i,:i], U[:i,j])
		
		for j in range(i+1,n):
			L[j,i] = (1/U[i,i]) * (A[j,i] - np.dot(L[j,:i], U[:i,i]))

	return L,U
	
	
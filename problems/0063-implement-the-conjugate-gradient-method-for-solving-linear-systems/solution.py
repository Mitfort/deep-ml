import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	x = np.zeros_like(b)
	r = residual(A, b, x)
	rPlus1 = r 
	p = r 

	for i in range(n):
		alp = alpha(A,r,p)

		x = x + alp * p 
		rPlus1 = r - alp * (A @ p)

		bet = beta(r,rPlus1)

		p = rPlus1 + bet * p 

		r = rPlus1

		if np.linalg.norm(residual(A,b,x)) < tol:
			break 
	
	return x


def residual(A:np.ndarray, b:np.ndarray, x:np.ndarray) -> np.ndarray:
	return b - A @ x 

def alpha(A:np.ndarray, r:np.ndarray, p:np.ndarray) -> float:
	alpha_num = np.dot(r,r)
	alpha_den = np.dot(p @ A, p)

	return alpha_num/alpha_den 

def beta(r:np.ndarray, r_plus1:np.ndarray) -> float: 
	beta_num = np.dot(r_plus1, r_plus1)
	beta_den = np.dot(r,r)

	return beta_num/beta_den 

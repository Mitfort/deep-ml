import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	
	x_np = np.array(x, dtype=float)
	f_x = np.array(f(x))

	n:int = len(x)
	m:int = len(f_x)

	J = np.ones((m,n))

	for j in range(n):
		x_cp = x_np.copy()

		x_cp[j] += h

		f_x_h = np.array(f(x_cp.tolist()))

		J[:,j] = (f_x_h - f_x) / h

	return J.tolist()
		
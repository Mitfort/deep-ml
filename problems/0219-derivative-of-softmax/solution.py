import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	
	max_x:float = np.max(x)
	exp = [np.exp(xi - max_x) for xi in x]

	sum_exp = np.sum(exp)

	s = [e / sum_exp for e in exp]

	n:int = len(s)
	J = [[0.0] * n for _ in range(n)]

	for i in range(n):
		for j in range(n):
			if i == j:
				J[i][j] = s[i] * (1 - s[i])
			else:
				J[i][j] = -s[i] * s[j]
	
	return J

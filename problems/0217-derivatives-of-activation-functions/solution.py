import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	relu = 1 if x > 0 else 0 

	sigmoid = 1 / (1 + np.exp(-x))
	sigmoid_der = sigmoid * (1 - sigmoid)

	tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	tanh_der = 1 - tanh**2

	return {"sigmoid": sigmoid_der.item(), "tanh": tanh_der.item(), "relu": relu}
import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	max_val = np.max(logits)
	exp = np.exp(logits - max_val)	# [exp(x1),exp(x2),..]
	
	exp_sum = np.sum(exp)

	# soft = (exp-exp_max) / exp_sum
	soft = (exp) / exp_sum
	
	# Subtract target 
	soft[target] -= 1

	return soft
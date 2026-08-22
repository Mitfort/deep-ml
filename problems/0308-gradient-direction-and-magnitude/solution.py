import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	gradient = np.array(gradient)
	
	if np.all(gradient == 0):
		zero = np.zeros_like(gradient)
		return {'magnitude':0, 'direction':zero, 'descent_direction':zero}

	l2_norm = np.sqrt(np.sum(gradient**2))

	direction = gradient / l2_norm

	return {'magnitude':l2_norm, 'direction':direction, 'descent_direction':-direction}
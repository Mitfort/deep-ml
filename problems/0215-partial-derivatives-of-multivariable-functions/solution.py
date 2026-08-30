import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	x = point[0]
	y = point[1]
	z = point[2] if len(point) == 3 else 0
	
	if func_name == "poly2d":
		dfdx = 2 * x * y + y**2
		dfdy = x**2 + 2 * x * y

		return (dfdx,dfdy)
	
	elif func_name == "exp_sum":
		sum_var = x + y
		dfdx = np.exp(sum_var) * (1 + y)
		dfdy = np.exp(sum_var) * (1 + x)

		return (dfdx,dfdy)
	
	elif func_name == "product_sin":
		dfdx = np.sin(y) + x * np.cos(y)
		dfdy = x * np.cos(y)

		return (dfdx,dfdy)

	elif func_name == "poly3d":
		dfdx = 2 * x * y
		dfdy = x**2 + z**2
		dfdz = 2 * y * z

		return (dfdx,dfdy,dfdz)
	
	elif func_name == "squared_error":
		dfdx = 2 * (x - y)
		dfdy = -2 * (x - y)

		return (dfdx,dfdy)
		
	return ()

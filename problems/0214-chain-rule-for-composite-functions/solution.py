import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""

	def func(val:float,func_name:str) -> float:
		if func_name == "square":
			return val**2
		
		if func_name == "sin":
			return np.sin(val)

		if func_name == "exp":
			return np.exp(val)
		
		if func_name == "log":
			return np.log(val)
		
		return 0

	def der(val:float, func_name):
		if func_name == "square":
			return 2*val

		if func_name == "sin":
			return np.cos(val)

		if func_name == "exp":
			return np.exp(val)

		if func_name == "log":
			return 1.0 / val

	#----
	n:int = len(functions)
	U:np.ndarray = np.ones(n+1)
	U[0] = x

	# XD to the logic right to left from functions 
	# FORWARD PASS 
	for i in range(n,0,-1):
		U[n-i+1] = func(U[n-i] ,functions[i-1])


	# BACKWARD
	der_x = 1.0

	for i in range(n, 0, -1):
		der_x *= der(U[n-i], functions[i-1])

	return der_x 



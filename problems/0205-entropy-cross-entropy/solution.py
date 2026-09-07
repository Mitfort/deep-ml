import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	
	P = np.array(P)
	Q = np.array(Q)
	eps = 1e-10

	H = -np.sum(P * np.log(P+eps))
	H2 = -np.sum(P * np.log(Q+eps))

	return (H,H2)
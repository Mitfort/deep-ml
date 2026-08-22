import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	if v1.shape != v2.shape:
		return None

	res:float = 0.0

	dot = np.sum(v1 * v2)

	l2_a = np.sqrt(np.sum(v1**2))
	l2_b = np.sqrt(np.sum(v2**2))

	res = dot / (l2_a * l2_b)

	return res



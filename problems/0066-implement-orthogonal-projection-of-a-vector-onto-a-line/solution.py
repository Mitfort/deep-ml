
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	
	dot_vl = sum([v[i] * L[i] for i in range(len(v))])
	dot_ll = sum([L[i]**2 for i in range(len(L))])

	dot = dot_vl / dot_ll

	proj = [L[i] * dot for i in range(len(L))]

	return proj

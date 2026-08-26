def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n:int = len(vectors)
	res = [[0] * n for _ in range(n)]

	def covariance(A:list[float], B:list[float]):
		len_a = len(A)
		len_b = len(B)
		
		mean_a = sum(A) / len_a
		mean_b = sum(B) / len_b

		cov=0.0

		for i in range(len_a):
			cov += (A[i] - mean_a) * (B[i] - mean_b)

		return cov / (len_a - 1)
	
	for i in range(n):
		for j in range(n):
			res[i][j] = covariance(vectors[i],vectors[j])

	return res
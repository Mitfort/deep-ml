def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	n,m = len(a), len(a[0])
	m_b = len(b)

	if m != m_b:
		return -1

	res = [0] * m
	
	for i in range(n):
		row_sum = 0.0
		for j in range(m):
			row_sum += a[i][j] * b[j]
		
		res[i] = row_sum

	return res
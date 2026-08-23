def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	rows,cols = len(matrix),len(matrix[0])

	means = None

	if mode == 'column':
		res = [0] * cols

		for col in range(cols):
			mean = 0.0
			for row in range(rows):
				mean += matrix[row][col]
			
			res[col] = mean / rows
		
		means = res

	elif mode == 'row':
		res = [0] * rows

		for row in range(rows):
			mean = 0.0
			for col in range(cols):
				mean += matrix[row][col]
			res[row] = mean / cols
		
		means = res 

	return means
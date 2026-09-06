def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a,b,c,d = float(matrix[0][0]), float(matrix[0][1]), float(matrix[1][0]), float(matrix[1][1])

    trace = a + d
    det = a * d - b * c

    delta = (trace**2 - 4*det)**(1/2)

    lam1 = (trace - delta) / 2
    lam2 = (trace + delta) / 2

    return [lam2,lam1]
	
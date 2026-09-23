import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.array(x,dtype=float)
    y = np.array(y,dtype=float)

    X = np.column_stack([x ** i for i in range(degree+1)])

    c = np.linalg.inv(X.T @ X) @ X.T @ y 

    return [float(coef) for coef in c]

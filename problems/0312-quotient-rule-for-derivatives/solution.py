import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    power_g = len(g_coeffs) - 1
    power_h = len(h_coeffs) - 1

    g_powers = np.arange(power_g, -1, -1)
    h_powers = np.arange(power_h, -1, -1)

    g_x = np.sum(g_coeffs * x**g_powers)
    h_x = np.sum(h_coeffs * x**h_powers)

    g_prim = np.sum(g_coeffs[:-1] * g_powers[:-1] * x**(g_powers[:-1] - 1))
    h_prim = np.sum(h_coeffs[:-1] * h_powers[:-1] * x**(h_powers[:-1] - 1))

    return (g_prim * h_x - g_x * h_prim) / h_x**2


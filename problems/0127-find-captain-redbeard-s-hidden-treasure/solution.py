def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    x = start_x
    grad = 4*(start_x**3) - 9 * (start_x**2)

    if abs(x) < 0.01:
        x = 0.1

    while grad:
        if abs(x) <= 0.01:
            x = 3

        x = x - 0.01 * grad 
        grad = 4*(x**3) - 9 * (x**2)

        if abs(grad) <= 0.0001:
            return x


    return x 

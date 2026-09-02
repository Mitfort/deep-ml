import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    
    e = np.eye(len(x))

    numerical_grad = [( f(x + epsilon * e[i]) - f(x - epsilon * e[i]) ) / (2 * epsilon) for i in range(len(x))]

    numerical_grad = np.array(numerical_grad)

    diff_norm = np.linalg.norm(numerical_grad - analytical_grad)
    grad_norm_sum = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)

    relative_error = diff_norm / max(1e-8, grad_norm_sum)
    
    return (numerical_grad, relative_error)
import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    
    # FORWARD PASS 
    z = W @ a_prev + b 
    a = 1 / (1 + np.exp(-z))

    # BACKWARD PASS 
    dC_da = 2 * (a - y)

    da_dz = a * (1 - a)

    dC_dz = dC_da * da_dz 

    dC_dw = np.outer(dC_dz, a_prev)

    dC_db = dC_dz 

    dC_da_prev = W.T @ dC_dz  

    return {
        'dW': dC_dw,
        'db': dC_db,
        'da_prev': dC_da_prev
    }




     

    

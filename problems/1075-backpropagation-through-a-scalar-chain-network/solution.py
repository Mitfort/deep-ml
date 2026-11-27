import numpy as np

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    w = np.array(w, dtype=float)
    b = np.array(b, dtype=float)
    z = np.zeros_like(w)
    a = np.zeros_like(w)
    prev = a0

    # FORWARD PASS 
    for i,(w_i,b_i) in enumerate(zip(w,b)):
        z[i] = w_i * prev + b_i  
        a[i] = 1 / (1 + np.exp(-z[i]))

        prev = a[i]

    residual = a[-1] - y

    dW = np.zeros_like(w)
    dB = np.zeros_like(b)
    d_prev = 2 * residual

    for i in range(len(z) - 1, -1, -1):
        a_prev = a[i-1] if i > 0 else a0 

        da_dz = a[i] * (1 - a[i])
        dC_dz = d_prev * da_dz

        dW[i] = dC_dz * a_prev
        dB[i] = dC_dz

        d_prev = dC_dz * w[i]

    dW = np.round(dW,6)
    dB = np.round(dB,6)

    return {
        'dW': dW.tolist(),
        'dB': dB.tolist()
    }


    


    
    

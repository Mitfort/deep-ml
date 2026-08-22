import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    
    vals: np.ndarray = arr.flatten()
    res:float = 0.0

    if norm_type == "l1":
        res = np.sum(np.abs(vals))
    elif norm_type == "l2":
        res = np.sqrt(np.sum(vals**2))
    elif norm_type == "frobenius":
        res = np.sqrt(np.sum(np.abs(vals)**2))
    
    return res
    
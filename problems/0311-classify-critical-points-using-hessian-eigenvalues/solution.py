import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    
    eigenvalues = np.linalg.eigvalsh(hessian)

    local_minimum = True
    local_maximum = True

    for eig in eigenvalues:
        if np.abs(eig) < tol: 
            return None 

        if eig < 0: 
            local_minimum = False
        
        if eig > 0:
            local_maximum = False

    if local_minimum:
        return -1
    elif local_maximum:
        return 1
    else:
        return 0 


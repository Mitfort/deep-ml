import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    if beta <= 0:
        raise ValueError("Beta must be grater than 0")

    y = np.array(target)
    y_hat = np.array(pred)

    err = (y - y_hat)

    loss = np.where(
        np.abs(err) < beta,
        0.5 * (err**2) / beta,
        np.abs(err) - 0.5 * beta
    )

    if reduction == 'mean':
        return loss.mean()
    
    elif reduction == "sum":
        return loss.sum()

    elif reduction == "none":
        return loss
    
    else:
        raise ValueError("Invalid reduction type")

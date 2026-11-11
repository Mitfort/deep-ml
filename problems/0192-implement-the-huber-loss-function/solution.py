def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    import numpy as np

    if not y_true or not y_pred: return 0.0

    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)

    n = len(y_true)

    mse = ((y_true - y_pred)**2 ) / 2 
    mae = np.abs(y_true - y_pred)

    h_loss = 0.0

    for i in range(n):
        if mae[i] <= delta:
            h_loss += mse[i]
        
        else:
            h_loss += delta * (mae[i] - delta / 2)
    
    return h_loss / n
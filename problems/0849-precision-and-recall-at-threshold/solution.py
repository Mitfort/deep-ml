import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    y_pred = y_scores >= threshold

    tp:int = np.sum((y_true == 1) & (y_pred == 1))
    fp:int = np.sum((y_true == 0) & (y_pred == 1))
    fn:int = np.sum((y_true == 1) & (y_pred == 0))

    precision = 0.0
    recall = 0.0 

    if np.sum(y_pred) != 0:
        precision = tp / (tp + fp)
    
    if np.sum(y_true) != 0:
        recall = tp / (tp + fn)

    return [round(precision,4),round(recall,4)]


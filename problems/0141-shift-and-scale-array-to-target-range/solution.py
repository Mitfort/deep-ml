import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """

    min_x = values.min()
    max_x = values.max()

    return c + (d - c) / (max_x - min_x) * (values - min_x)
    
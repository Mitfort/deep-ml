import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Create a copy to avoid mutating the original array in-place unexpectedly
    data = data.copy()
    _, COLS = data.shape
    
    for col in range(COLS):
        column_data = data[:, col]
        mask = np.isnan(column_data)
        
        # Skip columns that have no missing values or are entirely NaN
        if not np.any(mask) or np.all(mask):
            continue
            
        if strategy == "mean":
            replacement = np.nanmean(column_data)
        elif strategy == "median":
            replacement = np.nanmedian(column_data)
        elif strategy == "mode":
            # Compute mode for non-NaN values
            valid_data = column_data[~mask]
            vals, counts = np.unique(valid_data, return_counts=True)
            replacement = vals[np.argmax(counts)]
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Use 'mean', 'median', or 'mode'.")
            
        # Assign the replacement value to the missing indices in this column
        data[mask, col] = replacement

    return data
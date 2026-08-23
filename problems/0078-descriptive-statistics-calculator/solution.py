import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    n = len(data)
    mean = np.mean(data)
    median =  (data[n//2] + data[n//2-1])/2 if n % 2 == 0 else data[n//2]
    
    vals,counts = np.unique(data,return_counts=True)
    mode = vals[np.argmax(counts)]
    
    var = np.var(data)
    std = np.sqrt(var)

    per_25 = np.percentile(data,25)
    per_50 = np.percentile(data,50)
    per_75 = np.percentile(data,75)  

    iqr = per_75 - per_25

    return {
        'mean':mean,
        'median':median,
        'mode':mode,
        'variance':var, 
        'standard_deviation':std,
        '25th_percentile':per_25,
        '50th_percentile':per_50,
        '75th_percentile':per_75,
        'interquartile_range':iqr
    }
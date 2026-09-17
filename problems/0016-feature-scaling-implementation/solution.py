import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	
	means = data.mean(axis=0)
	stds = data.std(axis=0)

	standardized_data = (data - means) / stds


	mins = data.min(axis=0)
	maxes = data.max(axis=0)

	normalized_data = (data - mins) / (maxes - mins)

	return standardized_data, normalized_data
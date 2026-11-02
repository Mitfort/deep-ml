import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	data = np.asarray(data, dtype=float)
	
	Q1 = np.percentile(data, 25)
	Q2 = np.percentile(data, 75)

	IQR = Q2 - Q1 

	lower_bound = Q1 - k * IQR
	upper_bound = Q2 + k * IQR

	mask = (data > lower_bound) & (data < upper_bound)
	cleaned_data = data[mask]
	outlier_indices = np.where(~mask)[0]

	return {
		'cleaned_data': cleaned_data.tolist(),
		'outlier_indices': outlier_indices.tolist(),
		'lower_bound':float(lower_bound),
		'upper_bound':float(upper_bound)
	}
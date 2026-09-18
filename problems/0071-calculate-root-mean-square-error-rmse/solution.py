
import numpy as np

def rmse(y_true, y_pred):
	
	n:int = y_true.size

	rmse_res = np.sqrt(np.sum((y_true-y_pred)**2) / n)

	return round(rmse_res,3)

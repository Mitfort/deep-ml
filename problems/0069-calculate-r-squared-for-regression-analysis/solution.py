
import numpy as np

def r_squared(y_true, y_pred):
	
	SSR = np.sum((y_true - y_pred)**2)
	SST = np.sum((y_true - np.mean(y_true))**2)

	r2 = 1 - (SSR/SST)

	return r2

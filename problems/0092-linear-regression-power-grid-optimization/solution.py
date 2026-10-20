import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	x = np.array(consumption_data, dtype=np.float32)
	n:int = len(x)
	days = np.arange(1.0, 11.0, 1.0)
	days = np.float32(days)
	
	x -= 10 * np.sin(2*PI * days / 10)

	m = (n * np.sum(days * x) - np.sum(days) * np.sum(x)) / (n * np.sum(days**2) - np.sum(days)**2)

	b = (np.sum(x) - m * np.sum(days)) / n

	base_15 = m * 15 + b

	f_15 = 10 * np.sin(2*PI * 15 / 10)

	pred_15 = base_15 + f_15

	final_15 = 1.05 * round(pred_15)

	return math.ceil(final_15)

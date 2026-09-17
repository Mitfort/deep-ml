import numpy as np

def to_categorical(x, n_col=None):
	x = np.array(x, dtype=int)

	if n_col is None:
		n_col = np.max(x) + 1
        
	res = np.zeros((len(x), n_col), dtype=float)
    
	res[np.arange(len(x)), x] = 1
    
	return res

	# res = []

	# uniqe = np.unique(x)
	# n = len(uniqe)

	# for i in range(len(x)):
	# 	one_hot = [0] * n
	# 	one_hot[x[i]] = 1

	# 	res.append(one_hot)
	
	# return res
	
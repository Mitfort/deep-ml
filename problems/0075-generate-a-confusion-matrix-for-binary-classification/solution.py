
from collections import Counter

def confusion_matrix(data):
	
	ctr = Counter(tuple(pair) for pair in data)

	tp = ctr[(1,1)]
	tn = ctr[(0,0)]
	fn = ctr[(1,0)]
	fp = ctr[(0,1)]

	return [[tp,fn],[fp,tn]]

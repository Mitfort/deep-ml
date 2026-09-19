import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	
	losses: list[float] = []
	n,m = X.shape[0], X.shape[1]

	X = np.hstack([np.ones((X.shape[0], 1)), X])

	B = np.zeros(X.shape[1])

	for i in range(iterations):
		probs = 1 / (1 + np.exp(-X @ B))

		loss = -np.sum((y * np.log(probs) + (1-y) * np.log(1-probs)))
		losses.append(round(float(loss), 4))

		grad = X.T @ (probs - y)

		B = B - learning_rate * grad

	return [round(b,4) for b in B], losses
		

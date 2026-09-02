import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

	W:np.ndarray = initial_weights.copy()
	X:np.ndarray = features.copy()
	b:float = initial_bias

	losses:list[float] = []
	n_samples:int = X.shape[0]

	for epoch in range(epochs):
		z = W @ X.T + b

		pred = sigmoid(z)
		
		loss = np.mean((pred - labels)**2)

		losses.append(loss.item())

		dL_dz = (2 / n_samples) * (pred-labels) * (pred * (1-pred))

		dL_dW = dL_dz @ X
		dL_db = np.sum(dL_dz)

		W = W - learning_rate * dL_dW
		b = b - learning_rate * dL_db

	updated_weights = W
	updated_bias = b
	mse_values = losses

	return updated_weights, updated_bias, mse_values

def sigmoid(z:np.ndarray) -> np.ndarray:
	return 1 / (1 + np.exp(-z))
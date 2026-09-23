def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	
	updated_params = []

	for params, grads, apply_decay in zip(parameters,gradients,apply_to_all):
		updated_group = []
		for param, grad in zip(params,grads):
			if apply_decay:
				new_param = param - lr * grad - lr * weight_decay * param
			else:
				new_param = param - lr * grad

			updated_group.append(new_param)

		updated_params.append(updated_group)
	return updated_params
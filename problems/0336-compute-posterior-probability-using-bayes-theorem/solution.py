def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	
	numerators = [p * l for p,l in zip(priors,likelihoods)]

	P_E = sum(numerators)

	P_Hi_E = [num / P_E for num in numerators]

	return P_Hi_E
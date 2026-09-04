import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    dist = None
    mu = 0
    sigma = 0

    if distribution == "uniform":
        dist = np.random.uniform(0,1,size=(runs,n))
        mu = 0.5
        sigma = np.sqrt(1/12)
    elif distribution == "exponential":
        dist = np.random.exponential(1.0,size=(runs,n))
        mu = 1.0
        sigma = 1.0
    elif distribution == "bernoulli":
        dist = (np.random.rand(runs,n) < 0.3).astype(float)
        mu = 0.3
        sigma = np.sqrt(0.3 * 0.7)
    else:
        raise ValueError("Wrong dist")

    sample_means = dist.mean(axis=1)

    z = (sample_means - mu) / (sigma / np.sqrt(n) )

    return {'mean':z.mean(), 'std':z.std()}

    
    
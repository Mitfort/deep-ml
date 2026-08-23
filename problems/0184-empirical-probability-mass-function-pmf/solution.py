def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    
    counter = {}

    for val in samples:
        counter[val] = counter.get(val,0) + 1

    res = []
    n = len(samples)
    for key in counter.keys():
        prob = counter[key] / n
        res.append((key,prob))

    return res
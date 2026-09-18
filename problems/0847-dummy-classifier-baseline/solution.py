import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    y_train = np.array(y_train)
    ctr = Counter(y_train)
    classes = np.unique(y_train)

    if strategy == "most_frequent":
        max_freq = max(ctr.values())

        most_frequent_val = min(val for val, count in ctr.items() if count == max_freq)

        return [most_frequent_val] * n_test
    
    elif strategy == "constant":
        return [constant] * n_test
    
    elif strategy == "uniform":
        num_classes = len(classes)

        return [classes[i % num_classes] for i in range(n_test)]
        
    elif strategy == "stratified":
        total_samples = len(y_train)
        preds = []

        for i, c in enumerate(classes):
            prob =  ctr[c] / total_samples

            if i == len(classes) - 1:
                count = n_test - len(preds)
            else:
                count = int(round(n_test * prob))
            
            preds.extend([c] * count)


        return preds
        

        




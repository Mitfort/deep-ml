import numpy as np

def cross_features(X, levels, pairs):
    names = ["1"]
    for f, lv in enumerate(levels):
        names += [f"{f}={v}" for v in lv]
    for i, j in pairs:
        names += [f"{i}={u}&{j}={v}" for v in levels[j] for u in levels[i]]
    Phi = np.zeros((len(X), len(names)), dtype=float)
    col = {name: k for k, name in enumerate(names)}
    for n, row in enumerate(X):
        Phi[n, 0] = 1.0
        for f, lv in enumerate(levels):
            if row[f] in lv:
                Phi[n, col[f"{f}={row[f]}"]] = 1.0
        for i, j in pairs:
            if row[i] in levels[i] and row[j] in levels[j]:
                Phi[n, col[f"{i}={row[i]}&{j}={row[j]}"]] = 1.0
    return Phi, names

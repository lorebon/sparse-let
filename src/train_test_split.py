"""Reproducible stratified folds for interval sequences."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def stratified_indices(y, k=10, *, seed=1, shuffle=True):
    """Return train/test index pairs, requiring every class in every fold."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("The number of folds must be an integer of at least 2.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Classification requires at least two classes.")
    if counts.min() < k:
        raise ValueError(
            f"Each class needs at least {k} examples; smallest class has {counts.min()}."
        )
    splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed if shuffle else None)
    return list(splitter.split(np.zeros(len(y)), y))


def cross_validate(x, y, k, *, seed=1, shuffle=True):
    """Return (train_sequences, train_y, test_sequences, test_y) per fold."""
    if len(x) != len(y):
        raise ValueError("Sequences and labels must have the same length.")
    labels = np.asarray(y)
    return [
        ([x[i] for i in train], labels[train], [x[i] for i in test], labels[test])
        for train, test in stratified_indices(labels, k, seed=seed, shuffle=shuffle)
    ]

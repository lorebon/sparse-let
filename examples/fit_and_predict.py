"""Learn sparse-lets on AUSLAN2 and predict labels for held-out sequences.

Run with: python examples/fit_and_predict.py
"""

import sys
from pathlib import Path

import numpy as np
from numba import set_num_threads
from sklearn.model_selection import train_test_split

# Resolve imports from this file so the example works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abide import transform_sequences
from load_data import load_dataset
from sparselets import SearchConfig, fit_sparselets


def main():
    set_num_threads(1)
    seed = 1

    # 1. Load the data and hold out a quarter of each class for prediction.
    sequences, labels = load_dataset("AUSLAN2")
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, np.asarray(labels), test_size=0.25, stratify=labels, random_state=seed
    )

    # 2. Learn two sparse-lets using only the training sequences and labels.
    # AUSLAN2 sequences have 30 time rows, so a duration of 15 fits every example.
    config = SearchConfig(n_queries=2, generations=5, population_size=12, seed=seed)
    fitted = fit_sparselets(train_sequences, train_labels, duration=15, min_length=5, config=config)

    # 3. Match those same patterns to the held-out sequences and predict classes.
    test_features = transform_sequences(
        test_sequences, fitted.queries, method=config.method, distance=config.distance
    )
    predictions = fitted.classifier.predict(test_features)

    print(f"Training sequences: {len(train_sequences)}; test sequences: {len(test_sequences)}")
    print(f"Training features: {fitted.train_features.shape}; test features: {test_features.shape}")
    print(f"Test accuracy: {np.mean(predictions == test_labels):.3f}")
    print(f"First 10 predictions: {predictions[:10].tolist()}")
    print(f"First 10 true labels: {test_labels[:10].tolist()}")


if __name__ == "__main__":
    main()

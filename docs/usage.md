# Usage guide

[Overview and installation](../README.md) · [Datasets and input format](../data/README.md)

Run the commands below from the repository root with the environment activated.
`src/run_experiment.py` is the entry point for cross-validation experiments.

## Running experiments

```bash
# Repeatable generation-limited run, with full event alphabet
python src/run_experiment.py --dataset BLOCKS --folds 10 --queries 5 --generations 40 --seed 1

# Five-minute optimization budget per fold, using the paper's SVM choice
python src/run_experiment.py --dataset AUSLAN2 --folds 10 --queries 6 --time-limit 300 --length-fraction 0.5 --min-length-fraction 0.3

# Random forest with out-of-bag fitness
python src/run_experiment.py --dataset AUSLAN2 --classifier rf --fitness oob --distance squared --generations 40

# Experimental restriction to 80% of candidate event labels per query
python src/run_experiment.py --dataset AUSLAN2 --channel-fraction 0.8 --generations 40

python src/run_experiment.py --help
```

Use `--channel-fraction` to restrict the candidate event labels for each query.
Labels are sampled **without replacement**, once per query per fold. A fraction
of 1.0 uses the complete alphabet; 0.8 uses 80% of its labels, rounded to the
nearest integer with at least one label.

### Parameters

| Setting | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `AUSLAN2` | Bundled dataset, or a custom directory name |
| `--data-dir` | bundled `data/` | Parent directory for custom datasets |
| `--folds` | `10` | Seeded, shuffled, stratified cross-validation |
| `--queries` | `4` | Number of sparse-lets, K |
| `--length-fraction` | `0.5` | L as a fraction of the shortest sequence's table length |
| `--min-length-fraction` | `1/3` | Minimum retained interval length as a fraction of L |
| `--duration` / `--min-length` | derived from fractions | Set L and minimum interval length directly |
| `--alpha` | `0.00001` | Penalty per active event channel, summed over queries |
| `--classifier` / `--fitness` | `svm` / `train` | RBF SVM with in-sample fitness; `rf` / `oob` is also supported |
| `--svm-c` / `--trees` | `1.0` / `100` | SVM C or random-forest tree count |
| `--distance` | `euclidean` | Square root of binary mismatch count; `squared` uses the count itself |
| `--channel-fraction` | `1.0` | Fraction of candidate event labels per query |
| `--method` | `abide` | `reference`, `abide`, or `full`; all return the same exact distances |
| `--generations` | `40` | Used by default only when no time limit is supplied |
| `--time-limit` | unset | Optimization budget in seconds per fold |
| `--population-size` | chromosome length, at least 3 | Total of elites, offspring, and mutants |
| `--elite-fraction` / `--mutant-fraction` | `0.15` / `0.10` | BRKGA population shares, with at least one of each |
| `--bias` | `0.70` | Mating bias toward elite genes |
| `--seed` | `1` | Split seed; fold i uses seed + i − 1 for learning |
| `--threads` | `1` | Numba threads for sequence distances |
| `--output` | timestamped directory in `results/` | A new directory for result files |
| `--verbose` | off | Show optimizer progress |

`--duration` and `--min-length` set integer lengths directly and override their
respective fractions. Fractions are rounded to the nearest integer and clamped
to at least one. Every query must fit every evaluated sequence.

`--time-limit` is measured in seconds per fold and checked **between generations**;
a generation can run past the budget. Supplying both limits stops at whichever is
reached first. Numba warm-up is outside the optimization budget. Generation-limited
runs with identical data, settings, and dependency versions are repeatable;
time-limited runs may evaluate different numbers of candidates on different machines.

## How the method works

1. Convert each interval sequence into a binary time-by-event table.
2. Decode a chromosome into K sparse-lets. Each channel has a length gene and a
   start gene. Channels shorter than the minimum length are omitted; every
   retained channel contains at most one interval.
3. Slide each sparse-let over each training sequence. The minimum distance is one
   feature. Lower bounds and optional early abandonment accelerate this exact search.
4. Train a classifier on these K features and minimize
   `error / majority_class_accuracy + alpha * total_active_channels`.
5. Decode the best chromosome, fit a fresh classifier on the training embedding,
   and evaluate it on held-out sequences using the same learned patterns.

The default fitness uses training error, as described in the paper. It is an
optimization objective, not an estimate of test performance. `--fitness oob` uses
random-forest out-of-bag error instead. Held-out labels never enter feature learning
or classifier fitting. Following the paper's sliding-window constraint, the CLI
uses the shortest table length across the dataset to bound L, including test
sequence lengths.

A zero-valued channel is still part of the distance calculation: inactivity is
compared with the sequence. Sparse channels are not treated as missing or masked
features. An entirely inactive sparse-let is allowed and is saved explicitly.

## Fit and predict in Python

Run the [fit-and-predict example](../examples/fit_and_predict.py):

```bash
python examples/fit_and_predict.py
```

The example loads AUSLAN2, makes a stratified 75%/25% train/test split, learns two
sparse-lets on the training data, transforms the held-out sequences, and predicts
their labels. Its small search uses five generations and a population of 12.
Edit `SearchConfig`, `duration`, and `min_length` in the example to explore the method.
AUSLAN2 sequences have 30 time rows; the example uses L = 15 and Lmin = 5.

`fit_sparselets` returns a `FitResult` containing `queries`, `classifier`,
`train_features`, and search metadata. Pass the same distance and method settings
to `transform_sequences`, then pass those features to `fitted.classifier.predict`.
New sequences must use the same event-label mapping and have at least L time rows.

## Saved results and reuse

Each run writes:

- `summary.json`: settings, resolved lengths, seeds, dependency versions, input
  SHA-256 hashes, fold accuracy, objective, generation/evaluation counts, timing,
  active-channel counts, and final mean/standard deviation/pooled accuracy.
- `fold_XX.npz`: queries, chromosome, sampled channels, original train/test indices,
  both feature matrices, labels, and test predictions. All arrays can be read with
  `allow_pickle=False`.
- `fold_XX_queries.json`: readable intervals with original one-based event IDs.

The summary is updated after every completed fold. A stopped run retains completed
fold files and has `status: running`; automatic resume is not implemented. Fold
accuracy standard deviation is descriptive, not a confidence interval.

Load learned features from Python (run from the repository root):

```python
import sys
import numpy as np
sys.path.insert(0, "src")

from abide import transform_sequences
from load_data import load_dataset

sequences, labels = load_dataset("AUSLAN2")
with np.load("results/quickstart/fold_01.npz", allow_pickle=False) as fold:
    queries = fold["queries"]
    indices = fold["test_indices"]
    features = transform_sequences(
        [sequences[i] for i in indices], queries, distance="euclidean"
    )
    np.testing.assert_allclose(features, fold["test_features"])
```

Use the distance setting in the run's summary; `features_from_tables` and
`transform_sequences` default to `distance="squared"`. New sequences must
have the same event-label mapping and at least L time rows. This repository does
not pickle fitted classifiers. You can refit one from the saved training features,
labels, settings, and fold seed.

## Inspect saved queries

The readable `fold_XX_queries.json` files list each pattern's active intervals.
To plot a saved set of queries, use the existing plotting utility:

```bash
python src/plot_queries.py results/quickstart/fold_01.npz --output results/quickstart/queries.png
```

The output extension may be `.png`, `.pdf`, or `.svg`. Use `--columns` to change
the number of panels per row. Existing plot files are never overwritten.

## Runtime and memory

The implementation uses dense int32 event tables. All HEPATITIS tables together
occupy approximately 382 MiB before temporary copies; SKATING needs about 159 MiB.
Small K, a modest population, and a generation limit are useful while exploring.
More threads can help large datasets but add overhead for small ones. Classifier
fits are serial; there are no unmanaged multiprocessing pools.

## Development and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

Tests compare all distance implementations with an independent NumPy oracle,
exercise interval boundaries and malformed data, check the paper's decoder example,
verify repeatable SVM/RF runs, and inspect saved cross-validation outputs. GitHub
Actions is configured for Windows and Linux with Python 3.12.

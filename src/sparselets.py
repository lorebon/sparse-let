"""Sparse-let decoding and BRKGA learning shared by the experiment scripts."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import TerminateIfAny
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from abide import features_from_tables, preprocess_data


def decode_chromosome(x, duration, n_channels, n_queries, min_length, channel_subsets=None):
    """Decode pairs of length/start genes into fixed-duration binary sparse-lets.

    Each retained channel contains at most one interval. Round-to-nearest uses
    Python's ties-to-even rule. Leading/trailing zero rows are retained, as in
    the paper's fixed L representation. An all-zero sparse-let is valid.
    channel_subsets optionally restricts each query to sampled event labels.
    """
    for name, value in [
        ("duration", duration),
        ("n_channels", n_channels),
        ("n_queries", n_queries),
        ("min_length", min_length),
    ]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if min_length > duration:
        raise ValueError("min_length must not exceed duration.")
    if channel_subsets is None:
        channel_subsets = np.tile(np.arange(n_channels), (n_queries, 1))
    subsets = np.asarray(channel_subsets)
    if (
        subsets.ndim != 2
        or subsets.shape[0] != n_queries
        or subsets.shape[1] == 0
        or not np.issubdtype(subsets.dtype, np.integer)
        or np.any(subsets < 0)
        or np.any(subsets >= n_channels)
        or any(len(np.unique(row)) != len(row) for row in subsets)
    ):
        raise ValueError("Channel subsets require distinct valid channel indices per query.")
    genes = np.asarray(x, dtype=float)
    if (
        genes.ndim != 1
        or genes.size != 2 * subsets.size
        or not np.isfinite(genes).all()
        or np.any(genes < 0)
        or np.any(genes > 1)
    ):
        raise ValueError(f"Expected {2 * subsets.size} finite genes in [0, 1].")
    genes = genes.reshape(n_queries, subsets.shape[1], 2)
    queries = []
    for query_genes, channels in zip(genes, subsets):
        query = np.zeros((duration, n_channels), dtype=np.int32)
        for (length_gene, start_gene), channel in zip(query_genes, channels):
            length = round(length_gene * duration)
            if length >= min_length:
                start = round(start_gene * (duration - length))
                query[start : start + length, channel] = 1
        queries.append(query)
    return queries


def majority_accuracy(labels):
    """Fraction of observations in the largest class, independent of sample size."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    return float(np.unique(labels, return_counts=True)[1].max() / len(labels))


@dataclass(frozen=True)
class SearchConfig:
    """Explicit search settings; generation limits enable repeatable experiments."""

    n_queries: int = 4
    alpha: float = 1e-5
    classifier: str = "svm"
    fitness: str = "train"
    distance: str = "euclidean"
    method: str = "abide"
    generations: int | None = 40
    time_limit: float | None = None
    population_size: int | None = None
    elite_fraction: float = 0.15
    mutant_fraction: float = 0.10
    bias: float = 0.70
    seed: int = 1
    n_estimators: int = 100
    svm_c: float = 1.0
    channel_fraction: float = 1.0

    def __post_init__(self):
        for name in ("n_queries", "n_estimators"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.seed, (int, np.integer)) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if self.classifier not in ("svm", "rf") or self.fitness not in ("train", "oob"):
            raise ValueError("Use classifier svm/rf and fitness train/oob.")
        if self.fitness == "oob" and self.classifier != "rf":
            raise ValueError("Out-of-bag fitness requires --classifier rf.")
        if self.distance not in ("squared", "euclidean") or self.method not in (
            "reference",
            "abide",
            "full",
        ):
            raise ValueError("Unknown distance scale or method.")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1].")
        if not np.isfinite(self.svm_c) or self.svm_c <= 0:
            raise ValueError("svm_c must be positive and finite.")
        for name in ("elite_fraction", "mutant_fraction", "bias", "channel_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1].")
        if self.elite_fraction + self.mutant_fraction >= 1 or not 0.5 < self.bias < 1:
            raise ValueError("Elite + mutant fractions must be < 1; bias must lie in (0.5, 1).")
        if self.generations is None and self.time_limit is None:
            raise ValueError("Set a generation limit or a time limit.")
        if self.generations is not None and (
            not isinstance(self.generations, int) or self.generations < 1
        ):
            raise ValueError("generations must be a positive integer.")
        if self.time_limit is not None and (
            not np.isfinite(self.time_limit) or self.time_limit <= 0
        ):
            raise ValueError("time_limit must be positive and finite.")
        if self.population_size is not None and (
            not isinstance(self.population_size, int) or self.population_size < 3
        ):
            raise ValueError("population_size must be an integer of at least 3.")


def make_classifier(config, *, for_fitness=False):
    """Construct a fresh classifier; all stochastic estimators receive the seed."""
    if config.classifier == "svm":
        return SVC(C=config.svm_c, kernel="rbf", gamma="scale")
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.seed,
        n_jobs=1,
        oob_score=for_fitness and config.fitness == "oob",
        bootstrap=True,
    )


def evaluate_fitness(queries, tables, offsets, labels, config, *, validate=True):
    """Compute error / majority-class accuracy + alpha * total active channels.

    Only training labels enter this objective. OOB error is an optional RF
    alternative to the paper's in-sample error.
    """
    features = features_from_tables(
        tables,
        offsets,
        queries,
        method=config.method,
        distance=config.distance,
        validate=validate,
    )
    model = make_classifier(config, for_fitness=True)
    model.fit(features, labels)
    score = model.oob_score_ if config.fitness == "oob" else model.score(features, labels)
    active_channels = sum(np.count_nonzero(np.any(query, axis=0)) for query in queries)
    return float((1 - score) / majority_accuracy(labels) + config.alpha * active_channels)


class SparseLetProblem(ElementwiseProblem):
    """One BRKGA individual encodes the complete K-feature embedding."""

    def __init__(self, tables, offsets, labels, duration, min_length, subsets, config):
        self.tables, self.offsets, self.labels = tables, offsets, np.asarray(labels)
        self.duration, self.min_length = duration, min_length
        self.subsets, self.config = subsets, config
        super().__init__(n_var=2 * subsets.size, n_obj=1, xl=0.0, xu=1.0)

    def decode(self, x):
        return decode_chromosome(
            x,
            self.duration,
            self.tables.shape[1],
            self.config.n_queries,
            self.min_length,
            self.subsets,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = evaluate_fitness(
            self.decode(x),
            self.tables,
            self.offsets,
            self.labels,
            self.config,
            validate=False,
        )


@dataclass
class FitResult:
    queries: list
    classifier: object
    train_features: np.ndarray
    chromosome: np.ndarray
    channel_subsets: np.ndarray
    objective: float
    evaluations: int
    generations: int
    population_size: int
    elapsed_seconds: float


def fit_sparselets(sequences, labels, duration, min_length, config=None, *, verbose=False):
    """Learn queries and fit a final classifier using only the supplied training set."""
    config = SearchConfig() if config is None else config
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(sequences) or np.unique(labels).size < 2:
        raise ValueError("Provide one label per sequence and at least two classes.")
    tables, offsets = preprocess_data(sequences)
    if duration > np.min(np.diff(offsets)):
        raise ValueError("Query duration exceeds the shortest training sequence.")
    n_channels = tables.shape[1]
    subset_size = max(1, round(config.channel_fraction * n_channels))
    rng = np.random.default_rng(config.seed)
    subsets = (
        np.tile(np.arange(n_channels), (config.n_queries, 1))
        if subset_size == n_channels
        else np.stack(
            [rng.choice(n_channels, subset_size, replace=False) for _ in range(config.n_queries)]
        )
    )
    problem = SparseLetProblem(tables, offsets, labels, duration, min_length, subsets, config)
    # Validate dimensions and compile the kernel before the optimization timer starts.
    warmup_queries = problem.decode(np.zeros(problem.n_var))
    features_from_tables(
        tables[:duration],
        np.array([0, duration], dtype=np.int64),
        warmup_queries[:1],
        method=config.method,
        distance=config.distance,
    )
    population = config.population_size or max(3, problem.n_var)
    elites = max(1, int(config.elite_fraction * population))
    mutants = max(1, int(config.mutant_fraction * population))
    offspring = population - elites - mutants
    if offspring < 1:
        raise ValueError("Population is too small for the elite and mutant fractions.")
    algorithm = BRKGA(n_elites=elites, n_offsprings=offspring, n_mutants=mutants, bias=config.bias)
    criteria = []
    if config.generations is not None:
        criteria.append(MaximumGenerationTermination(config.generations))
    if config.time_limit is not None:
        criteria.append(TimeBasedTermination(config.time_limit))
    start = perf_counter()
    result = minimize(
        problem, algorithm, TerminateIfAny(*criteria), seed=config.seed, verbose=verbose
    )
    if result.X is None:
        raise RuntimeError("The optimizer returned no solution.")
    queries = problem.decode(result.X)
    features = features_from_tables(
        tables,
        offsets,
        queries,
        method=config.method,
        distance=config.distance,
        validate=False,
    )
    classifier = make_classifier(config)
    classifier.fit(features, labels)
    return FitResult(
        queries,
        classifier,
        features,
        result.X,
        subsets,
        float(np.asarray(result.F).item()),
        int(result.algorithm.evaluator.n_eval),
        int(result.algorithm.n_gen - 1),
        population,
        perf_counter() - start,
    )

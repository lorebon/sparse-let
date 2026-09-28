"""Command-line cross-validation for Sparse-let."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numba import config as numba_config
from numba import set_num_threads

from abide import transform_sequences
from load_data import DATA_DIR, DATASETS, load_dataset
from sparselets import SearchConfig, fit_sparselets
from train_test_split import stratified_indices


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="AUSLAN2",
        help=f"Dataset directory name; bundled: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="Parent directory of datasets"
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified folds")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Split seed; each fold uses seed + fold index",
    )
    parser.add_argument("--queries", type=int, default=4, help="Number of sparse-lets (K)")
    parser.add_argument("--duration", type=int, help="Explicit sparse-let duration (L)")
    parser.add_argument(
        "--length-fraction",
        type=float,
        default=0.5,
        help="Fraction of shortest sequence if L is not explicit",
    )
    parser.add_argument("--min-length", type=int, help="Explicit minimum retained interval length")
    parser.add_argument(
        "--min-length-fraction",
        type=float,
        default=1 / 3,
        help="Fraction of L if minimum is not explicit",
    )
    parser.add_argument("--alpha", type=float, default=1e-5, help="Active-channel penalty")
    parser.add_argument("--classifier", choices=["svm", "rf"], default="svm")
    parser.add_argument(
        "--fitness",
        choices=["train", "oob"],
        default="train",
        help="OOB is only available for RF",
    )
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=100, help="RF tree count")
    parser.add_argument("--distance", choices=["euclidean", "squared"], default="euclidean")
    parser.add_argument("--method", choices=["reference", "abide", "full"], default="abide")
    parser.add_argument(
        "--generations",
        type=int,
        help="Generation limit; defaults to 40 unless time limit is given",
    )
    parser.add_argument(
        "--time-limit", type=float, help="Seconds per fold, checked between generations"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        help="Total population; defaults to chromosome length (at least 3)",
    )
    parser.add_argument("--elite-fraction", type=float, default=0.15)
    parser.add_argument("--mutant-fraction", type=float, default=0.10)
    parser.add_argument("--bias", type=float, default=0.70)
    parser.add_argument(
        "--channel-fraction",
        type=float,
        default=1.0,
        help="Fraction of labels sampled without replacement for each query",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Numba distance threads (no worker processes)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New output directory; existing directories are never overwritten",
    )
    parser.add_argument("--verbose", action="store_true", help="Show optimizer progress")
    return parser


def query_intervals(queries):
    """Human-readable retained intervals; event IDs match the one-based input IDs."""
    records = []
    for index, query in enumerate(queries):
        intervals = []
        for channel in range(query.shape[1]):
            active = np.flatnonzero(query[:, channel])
            if len(active):
                intervals.append(
                    {
                        "event_id": channel + 1,
                        "start": int(active[0]),
                        "end": int(active[-1] + 1),
                    }
                )
        records.append({"query": index, "duration": len(query), "intervals": intervals})
    return records


def _write_json(path, data):
    # Replace atomically so a stopped run still has a readable last-completed summary.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_experiment(args):
    """Run all folds and save settings, learned queries, embeddings and predictions."""
    generations = args.generations
    if generations is None and args.time_limit is None:
        generations = 40
    search = SearchConfig(
        n_queries=args.queries,
        alpha=args.alpha,
        classifier=args.classifier,
        fitness=args.fitness,
        distance=args.distance,
        method=args.method,
        generations=generations,
        time_limit=args.time_limit,
        population_size=args.population_size,
        elite_fraction=args.elite_fraction,
        mutant_fraction=args.mutant_fraction,
        bias=args.bias,
        seed=args.seed,
        n_estimators=args.trees,
        svm_c=args.svm_c,
        channel_fraction=args.channel_fraction,
    )
    if not 1 <= args.threads <= numba_config.NUMBA_NUM_THREADS:
        raise ValueError(f"threads must lie in [1, {numba_config.NUMBA_NUM_THREADS}].")
    for name in ("length_fraction", "min_length_fraction"):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"{name} must lie in (0, 1].")
    if args.seed + args.folds >= 2**32:
        raise ValueError("seed + folds must be smaller than 2**32.")
    set_num_threads(args.threads)
    sequences, labels = load_dataset(args.dataset, data_dir=args.data_dir)
    labels = np.asarray(labels)
    folds = stratified_indices(labels, args.folds, seed=args.seed)
    # The paper constrains L using all sequence lengths, but no test class labels
    # or test embeddings enter the optimizer or the final classifier fit.
    lengths = [
        max(
            (end for channel in sequence for start, end, *_ in channel if start < end),
            default=0,
        )
        for sequence in sequences
    ]
    shortest = min(lengths)
    if shortest < 1:
        raise ValueError("Every sequence needs at least one positive-duration interval.")
    duration = (
        args.duration
        if args.duration is not None
        else max(1, round(shortest * args.length_fraction))
    )
    min_length = (
        args.min_length
        if args.min_length is not None
        else max(1, round(duration * args.min_length_fraction))
    )
    if not 1 <= duration <= shortest or not 1 <= min_length <= duration:
        raise ValueError("Require 1 <= min_length <= duration <= shortest sequence length.")
    dataset = args.dataset.upper()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("results") / f"{dataset.lower()}-{timestamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    directory = args.data_dir.resolve() / dataset
    summary = {
        "schema_version": 1,
        "status": "running",
        "dataset": dataset,
        "created_utc": timestamp,
        "data_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("data.txt", "classes.txt")
        },
        "n_sequences": len(sequences),
        "n_channels": len(sequences[0]),
        "n_classes": len(np.unique(labels)),
        "duration": duration,
        "min_length": min_length,
        "n_folds": args.folds,
        "split_seed": args.seed,
        "shuffle": True,
        "threads": args.threads,
        "search": asdict(search),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "numba", "scikit-learn", "pymoo")},
        },
        "folds": [],
    }
    _write_json(output / "summary.json", summary)
    print(
        f"{dataset}: {len(sequences)} sequences, {len(sequences[0])} channels; L={duration}, Lmin={min_length}",
        flush=True,
    )
    print(f"Results: {output}", flush=True)
    for index, (train, test) in enumerate(folds):
        fold_config = replace(search, seed=args.seed + index)
        fitted = fit_sparselets(
            [sequences[i] for i in train],
            labels[train],
            duration,
            min_length,
            fold_config,
            verbose=args.verbose,
        )
        test_features = transform_sequences(
            [sequences[i] for i in test],
            fitted.queries,
            method=search.method,
            distance=search.distance,
        )
        predictions = fitted.classifier.predict(test_features)
        accuracy = float(np.mean(predictions == labels[test]))
        stem = f"fold_{index + 1:02d}"
        np.savez_compressed(
            output / f"{stem}.npz",
            queries=np.stack(fitted.queries),
            chromosome=fitted.chromosome,
            channel_subsets=fitted.channel_subsets,
            train_indices=train,
            test_indices=test,
            train_features=fitted.train_features,
            test_features=test_features,
            train_labels=labels[train],
            test_labels=labels[test],
            predictions=predictions,
        )
        _write_json(output / f"{stem}_queries.json", query_intervals(fitted.queries))
        summary["folds"].append(
            {
                "fold": index + 1,
                "seed": fold_config.seed,
                "accuracy": accuracy,
                "objective": fitted.objective,
                "evaluations": fitted.evaluations,
                "generations": fitted.generations,
                "population_size": fitted.population_size,
                "fit_seconds": fitted.elapsed_seconds,
                "active_channels_per_query": [int(np.any(q, axis=0).sum()) for q in fitted.queries],
                "arrays": f"{stem}.npz",
                "queries": f"{stem}_queries.json",
            }
        )
        _write_json(output / "summary.json", summary)
        print(
            f"Fold {index + 1}/{len(folds)}: accuracy={accuracy:.4f}, objective={fitted.objective:.6f}, fit={fitted.elapsed_seconds:.1f}s",
            flush=True,
        )
    scores = [fold["accuracy"] for fold in summary["folds"]]
    summary.update(
        status="complete",
        mean_accuracy=float(np.mean(scores)),
        std_accuracy=float(np.std(scores)),
        pooled_accuracy=float(np.average(scores, weights=[len(test) for _, test in folds])),
    )
    _write_json(output / "summary.json", summary)
    print(
        f"Mean accuracy: {summary['mean_accuracy']:.4f} (fold SD {summary['std_accuracy']:.4f})",
        flush=True,
    )
    return output


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_experiment(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

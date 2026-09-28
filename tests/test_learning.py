import json

import numpy as np
import pytest

from abide import preprocess_data, transform_sequences
from run_experiment import build_parser, query_intervals, run_experiment
from sparselets import (
    SearchConfig,
    decode_chromosome,
    evaluate_fitness,
    fit_sparselets,
    majority_accuracy,
)


def toy_data():
    sequences = []
    for index in range(12):
        sequences.append([[[0, 2]], [[2, 4]]] if index < 6 else [[[2, 4]], [[0, 2]]])
    return sequences, np.repeat([0, 1], 6)


def test_paper_figure_3_decodes_without_truncation():
    genes = [0.57, 0.11, 0.08, 0.75, 0.34, 0.68, 0.29, 0.95, 0.20, 0.41]
    query = decode_chromosome(genes, 7, 5, 1, 2)[0]
    expected = np.zeros((7, 5), dtype=np.int32)
    expected[0:4, 0] = 1
    expected[3:5, 2] = 1
    expected[5:7, 3] = 1
    np.testing.assert_array_equal(query, expected)
    assert query_intervals([query])[0]["intervals"] == [
        {"event_id": 1, "start": 0, "end": 4},
        {"event_id": 3, "start": 3, "end": 5},
        {"event_id": 4, "start": 5, "end": 7},
    ]


def test_decoder_boundaries_and_empty_queries():
    np.testing.assert_array_equal(decode_chromosome([1, 1], 5, 1, 1, 1)[0], np.ones((5, 1)))
    assert not decode_chromosome([0, 1], 5, 1, 1, 1)[0].any()
    assert decode_chromosome([0.4, 1], 5, 1, 1, 2)[0][:, 0].tolist() == [0, 0, 0, 1, 1]
    query = decode_chromosome([1, 1], 5, 3, 1, 3, channel_subsets=np.array([[2]]))[0]
    np.testing.assert_array_equal(query.sum(axis=0), [0, 0, 5])


@pytest.mark.parametrize("genes", [[0], [0, np.nan], [-1, 0], [2, 0]])
def test_invalid_genes(genes):
    with pytest.raises(ValueError):
        decode_chromosome(genes, 5, 1, 1, 1)


def test_invalid_subsets():
    with pytest.raises(ValueError):
        decode_chromosome([1, 0, 1, 0], 4, 2, 1, 1, [[0, 0]])


def test_majority_accuracy_is_a_fraction_and_sample_size_invariant():
    assert majority_accuracy([0, 0, 0, 1]) == 0.75
    assert majority_accuracy([0, 0, 0, 1] * 10) == 0.75


def test_fitness_normalization_and_regularization():
    sequences, _ = toy_data()
    labels = np.array([0] * 9 + [1] * 3)
    tables, offsets = preprocess_data(sequences)
    # A fully inactive query gives identical features for every toy sequence.
    queries = [np.zeros((4, 2), dtype=np.int32)]
    config = SearchConfig(alpha=0.1)
    assert evaluate_fitness(queries, tables, offsets, labels, config) == pytest.approx(0.25 / 0.75)
    queries[0][:, 0] = 1
    assert evaluate_fitness(queries, tables, offsets, labels, config) == pytest.approx(
        0.25 / 0.75 + 0.1
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"classifier": "svm", "fitness": "oob"},
        {"generations": 0},
        {"time_limit": -1},
        {"alpha": float("nan")},
        {"n_queries": 0},
        {"channel_fraction": 0},
        {"elite_fraction": 0.8, "mutant_fraction": 0.5},
        {"population_size": 2},
    ],
)
def test_invalid_search_settings(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)


@pytest.mark.parametrize("classifier,fitness,fraction", [("svm", "train", 1.0), ("rf", "oob", 0.5)])
def test_learning_is_repeatable_and_population_is_total(classifier, fitness, fraction):
    sequences, labels = toy_data()
    config = SearchConfig(
        n_queries=2,
        classifier=classifier,
        fitness=fitness,
        generations=2,
        population_size=8,
        seed=19,
        channel_fraction=fraction,
    )
    first = fit_sparselets(sequences, labels, 4, 1, config)
    second = fit_sparselets(sequences, labels, 4, 1, config)
    assert first.population_size == 8
    assert first.generations == 2
    assert first.evaluations == 15  # 8 initial + 6 offspring + 1 mutant.
    assert first.objective == second.objective
    np.testing.assert_array_equal(first.chromosome, second.chromosome)
    np.testing.assert_array_equal(first.queries, second.queries)
    np.testing.assert_array_equal(
        first.train_features,
        transform_sequences(sequences, first.queries, distance=config.distance),
    )
    assert first.channel_subsets.shape == (2, round(fraction * 2))


@pytest.mark.parametrize("channel_fraction", [1.0, 0.5])
def test_cli_artifacts_and_no_overwrite(tmp_path, channel_fraction):
    directory = tmp_path / "data" / "TOY"
    directory.mkdir(parents=True)
    sequences, labels = toy_data()
    rows = []
    for index, sequence in enumerate(sequences):
        for channel, intervals in enumerate(sequence, 1):
            for start, end in intervals:
                rows.append(f"{index} {channel} {start} {end}\n")
    (directory / "data.txt").write_text("".join(rows))
    (directory / "classes.txt").write_text("\n".join(map(str, labels)))
    output = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--dataset",
            "TOY",
            "--data-dir",
            str(directory.parent),
            "--folds",
            "2",
            "--generations",
            "1",
            "--population-size",
            "6",
            "--queries",
            "2",
            "--channel-fraction",
            str(channel_fraction),
            "--output",
            str(output),
        ]
    )
    run_experiment(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert len(summary["folds"]) == 2
    all_test = []
    for fold in summary["folds"]:
        with np.load(output / fold["arrays"], allow_pickle=False) as arrays:
            assert arrays["queries"].shape == (2, 2, 2)
            assert arrays["channel_subsets"].shape == (2, round(2 * channel_fraction))
            for query, channels in zip(arrays["queries"], arrays["channel_subsets"]):
                excluded = np.setdiff1d(np.arange(query.shape[1]), channels)
                assert not query[:, excluded].any()
            np.testing.assert_allclose(
                fold["accuracy"], np.mean(arrays["predictions"] == arrays["test_labels"])
            )
            assert not set(arrays["train_indices"]) & set(arrays["test_indices"])
            all_test.extend(arrays["test_indices"])
            np.testing.assert_array_equal(arrays["test_labels"], labels[arrays["test_indices"]])
            np.testing.assert_allclose(
                arrays["test_features"],
                transform_sequences(
                    [sequences[i] for i in arrays["test_indices"]],
                    arrays["queries"],
                    distance="euclidean",
                ),
            )
    assert sorted(all_test) == list(range(12))
    with pytest.raises(FileExistsError):
        run_experiment(args)


def test_time_only_termination():
    sequences, labels = toy_data()
    fitted = fit_sparselets(
        sequences,
        labels,
        2,
        1,
        SearchConfig(n_queries=1, population_size=4, generations=None, time_limit=0.000001),
    )
    assert fitted.generations == 1

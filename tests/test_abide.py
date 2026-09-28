"""Numerical checks against an independent NumPy sliding-window oracle."""

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from abide import (
    abide,
    abide_full,
    abide_old,
    features_from_tables,
    get_event_table,
    preprocess_data,
)


def oracle(tables, query):
    return np.array(
        [
            min(
                np.count_nonzero(table[start : start + len(query)] != query)
                for start in range(len(table) - len(query) + 1)
            )
            for table in tables
        ]
    )


def pack(tables):
    return np.concatenate(tables), np.r_[0, np.cumsum([len(t) for t in tables])]


@pytest.mark.parametrize("function", [abide_old, abide, abide_full])
def test_sequence_boundaries_and_last_window(function):
    query = np.array([[1], [1]], dtype=np.int32)
    tables = [
        query.copy(),
        np.zeros((2, 1), dtype=np.int32),
        np.array([[0], [0], [1], [1]], dtype=np.int32),
    ]
    values, offsets = pack(tables)
    np.testing.assert_array_equal(function(values, offsets, query), [0, 2, 0])


@pytest.mark.parametrize("seed", range(12))
def test_all_pruning_methods_equal_independent_oracle(seed):
    rng = np.random.default_rng(seed)
    channels = int(rng.integers(1, 9))
    tables = [rng.integers(0, 2, (length, channels), dtype=np.int32) for length in (8, 13, 22)]
    values, offsets = pack(tables)
    queries = [rng.integers(0, 2, (length, channels), dtype=np.int32) for length in (1, 3, 8)]
    queries.extend(
        [np.zeros((4, channels), dtype=np.int32), np.ones((4, channels), dtype=np.int32)]
    )
    expected = np.column_stack([oracle(tables, query) for query in queries])
    for method in ("reference", "abide", "full"):
        actual = features_from_tables(values, offsets, queries, method=method)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        features_from_tables(values, offsets, queries, distance="euclidean"), np.sqrt(expected)
    )


def test_parallel_scratch_arrays_are_independent():
    rng = np.random.default_rng(77)
    tables = [rng.integers(0, 2, (30, 5), dtype=np.int32) for _ in range(20)]
    query = rng.integers(0, 2, (7, 5), dtype=np.int32)
    values, offsets = pack(tables)
    previous = get_num_threads()
    try:
        set_num_threads(1)
        serial = abide_full(values, offsets, query)
        set_num_threads(previous)
        np.testing.assert_array_equal(abide_full(values, offsets, query), serial)
    finally:
        set_num_threads(previous)


def test_event_table_unions_unsorted_adjacent_overlapping_intervals():
    actual = get_event_table([[[4, 6], [1, 3], [3, 4], [2, 5]], [[2, 2]], []])
    expected = np.zeros((6, 3), dtype=np.int32)
    expected[1:6, 0] = 1
    np.testing.assert_array_equal(actual, expected)
    assert get_event_table([[], [[9, 9]]]).shape == (0, 2)


@pytest.mark.parametrize("interval", [[-1, 2], [3, 2], [0.5, 3], [0, float("nan")]])
def test_bad_timestamps_raise(interval):
    with pytest.raises(ValueError):
        get_event_table([[interval]])


@pytest.mark.parametrize(
    "queries", [[], [np.ones((4, 1))], [np.ones((0, 1))], [np.ones((1, 2))], [np.array([[2]])]]
)
def test_invalid_queries_raise(queries):
    with pytest.raises(ValueError):
        features_from_tables(np.ones((3, 1)), np.array([0, 3]), queries)


@pytest.mark.parametrize("offsets", [[1, 3], [0, 0, 3], [0, 4], [0.0, 3.0]])
def test_invalid_offsets_raise(offsets):
    with pytest.raises(ValueError):
        abide(np.ones((3, 1)), offsets, np.ones((1, 1)))


@pytest.mark.parametrize("sequences", [[], [[[]]], [[[[0, 1]]], [[], []]]])
def test_invalid_sequences_raise(sequences):
    with pytest.raises(ValueError):
        preprocess_data(sequences)

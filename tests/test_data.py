import numpy as np
import pytest

from load_data import DATASETS, load_dataset, read_sequences
from train_test_split import cross_validate, stratified_indices
from transform_to_input_format import transform_to_input_format


@pytest.mark.parametrize(
    "name,size,channels,classes",
    [
        ("AUSLAN2", 200, 12, 10),
        ("BLOCKS", 210, 8, 8),
        ("CONTEXT", 240, 54, 5),
        ("HEPATITIS", 498, 63, 2),
        ("PIONEER", 160, 92, 3),
        ("SKATING", 530, 41, 6),
    ],
)
def test_bundled_data_from_another_directory(name, size, channels, classes, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    x, y = load_dataset(name.lower())
    assert name in DATASETS
    assert len(x) == len(y) == size
    assert all(len(sequence) == channels for sequence in x)
    assert len(set(y)) == classes


def test_parser_whitespace_sorting_and_shared_alphabet(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("0\t2  3.0 4\n\n1 1 0 2 0.5\n0 2 1 3\n")
    x = read_sequences(source, cast_to_int=True)
    assert x == [[[], [[1, 3, 1.0], [3, 4, 1.0]]], [[[0, 2, 0.5]], []]]


@pytest.mark.parametrize(
    "row", ["0 0 0 1", "0 1 2 1", "0 1 nan 1", "0 1 0.5 1", "0 1 0", "0 1 0 1 2 3", "-1 1 0 1"]
)
def test_malformed_rows_report_path_and_line(row, tmp_path):
    source = tmp_path / "bad.txt"
    source.write_text(row)
    with pytest.raises(ValueError, match=r"bad.txt:1:"):
        read_sequences(source, cast_to_int=True)


def test_missing_sequence_and_label_mismatch(tmp_path):
    directory = tmp_path / "CUSTOM"
    directory.mkdir()
    source = directory / "data.txt"
    source.write_text("1 1 0 1\n")
    with pytest.raises(ValueError, match="contiguous"):
        read_sequences(source)
    source.write_text("0 1 0 1\n1 1 0 2\n")
    (directory / "classes.txt").write_text("1\n")
    with pytest.raises(ValueError, match="2 sequences but 1"):
        load_dataset("custom", data_dir=tmp_path)


def test_fold_partition_is_reproducible_stratified_and_disjoint():
    labels = np.repeat([4, 8, 9], [12, 15, 18])
    folds = stratified_indices(labels, 3, seed=11)
    repeated = stratified_indices(labels, 3, seed=11)
    test_indices = []
    for (train, test), (train2, test2) in zip(folds, repeated):
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(len(labels)))
        np.testing.assert_array_equal(np.unique(labels[test], return_counts=True)[1], [4, 5, 6])
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(test, test2)
        test_indices.extend(test)
    assert sorted(test_indices) == list(range(len(labels)))
    assert any(
        not np.array_equal(test, other)
        for (_, test), (_, other) in zip(folds, stratified_indices(labels, 3, seed=12))
    )


@pytest.mark.parametrize("labels,k", [([0, 0], 2), ([0, 1], 2), ([0, 0, 1, 1], 1)])
def test_invalid_folds(labels, k):
    with pytest.raises(ValueError):
        stratified_indices(labels, k)
    with pytest.raises(ValueError):
        cross_validate([], labels, 2)


def test_costi_converter_keeps_signed_endpoints_and_empty_examples():
    times, channels, values, offsets = transform_to_input_format(
        [[(0.5, 2.5, 1, 3), (4, 4, 0, 1)], [], [(1, 2, 0, -1)]]
    )
    np.testing.assert_array_equal(times, [0.5, 2.5, 1, 2])
    np.testing.assert_array_equal(channels, [1, 1, 0, 0])
    np.testing.assert_array_equal(values, [3, -3, -1, 1])
    np.testing.assert_array_equal(offsets, [0, 2, 2, 4])

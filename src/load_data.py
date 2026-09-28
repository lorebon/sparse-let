"""Read interval datasets independently of the current working directory."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("AUSLAN2", "BLOCKS", "CONTEXT", "HEPATITIS", "PIONEER", "SKATING")


def read_sequences(file_path, cast_to_int=False):
    """Read whitespace-separated sequence_id event_id start end [value] rows.

    Sequence IDs are contiguous from zero; event IDs start at one. Return
    sequences[sequence][channel][interval], with [start, end, value] triples.
    All sequences share an alphabet. cast_to_int requires integral timestamps
    instead of silently truncating fractional times.
    """
    path = Path(file_path)
    sequences, max_events = {}, 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                if len(fields) not in (4, 5):
                    raise ValueError("expected 4 or 5 fields")
                sequence_id, event_id = int(fields[0]), int(fields[1])
                start, end = float(fields[2]), float(fields[3])
                value = float(fields[4]) if len(fields) == 5 else 1.0
                if sequence_id < 0 or event_id < 1:
                    raise ValueError("sequence IDs start at 0 and event IDs at 1")
                if not np.isfinite([start, end, value]).all() or not 0 <= start <= end:
                    raise ValueError("expected finite values and 0 <= start <= end")
                if cast_to_int:
                    if not start.is_integer() or not end.is_integer():
                        raise ValueError("integer event tables require integral timestamps")
                    start, end = int(start), int(end)
                channels = sequences.setdefault(sequence_id, {})
                channels.setdefault(event_id - 1, []).append([start, end, value])
                max_events = max(max_events, event_id)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not sequences:
        raise ValueError(f"{path}: no intervals found")
    if sorted(sequences) != list(range(len(sequences))):
        raise ValueError(f"{path}: sequence IDs must be contiguous from zero")
    return [
        [
            sorted(sequences[i].get(c, []), key=lambda event: (event[0], event[1]))
            for c in range(max_events)
        ]
        for i in range(len(sequences))
    ]


def read_classes(file_path):
    """Read one integer class label per nonblank line."""
    path, classes = Path(file_path), []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    classes.append(int(line))
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: expected an integer class label"
                    ) from error
    if not classes:
        raise ValueError(f"{path}: no class labels found")
    return classes


def load_dataset(name, cast_to_int=True, data_dir=None):
    """Load data_dir/NAME/{data,classes}.txt and check label alignment.

    Custom names are supported; names are normalized to uppercase.
    """
    name = str(name).upper()
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Dataset name must be a single directory name.")
    directory = (DATA_DIR if data_dir is None else Path(data_dir)) / name
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    sequences = read_sequences(directory / "data.txt", cast_to_int=cast_to_int)
    classes = read_classes(directory / "classes.txt")
    if len(sequences) != len(classes):
        raise ValueError(f"{directory}: {len(sequences)} sequences but {len(classes)} class labels")
    return sequences, classes


# Original convenience functions for the six bundled datasets.
def load_auslan2(cast_to_int=False):
    return load_dataset("AUSLAN2", cast_to_int)


def load_blocks(cast_to_int=False):
    return load_dataset("BLOCKS", cast_to_int)


def load_context(cast_to_int=False):
    return load_dataset("CONTEXT", cast_to_int)


def load_hepatitis(cast_to_int=False):
    return load_dataset("HEPATITIS", cast_to_int)


def load_pioneer(cast_to_int=False):
    return load_dataset("PIONEER", cast_to_int)


def load_skating(cast_to_int=False):
    return load_dataset("SKATING", cast_to_int)


def get_all_old_methods():
    return [
        load_auslan2,
        load_blocks,
        load_context,
        load_hepatitis,
        load_pioneer,
        load_skating,
    ]

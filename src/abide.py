"""Binary event tables and exact sliding-window distances.

Rows represent unit intervals [t, t + 1); columns represent event labels.
The distance kernels return mismatch counts (squared Euclidean distances
for binary tables). The feature helpers also support Euclidean distances.
"""

import numpy as np
from numba import njit, prange


def get_event_table(channels):
    """Convert channels of (start, end[, value]) intervals to a binary table.

    Timestamps must be nonnegative integers. Overlapping, adjacent and unsorted
    intervals are supported. Instantaneous events are ignored; values do not
    affect binary activity. Empty input gives shape (0, n_channels). The time
    origin is zero, preserving leading inactivity in the input.
    """
    intervals = []
    end_time = 0
    for channel, events in enumerate(channels):
        for event in events:
            if len(event) not in (2, 3):
                raise ValueError("An interval requires start, end and optionally value.")
            start, end = event[:2]
            if not np.isfinite(start) or not np.isfinite(end):
                raise ValueError("Interval timestamps must be finite.")
            if int(start) != start or int(end) != end or start < 0 or end < start:
                raise ValueError("Intervals require integers with 0 <= start <= end.")
            if start != end:
                intervals.append((int(start), int(end), channel))
                end_time = max(end_time, int(end))
    table = np.zeros((end_time, len(channels)), dtype=np.int32)
    for start, end, channel in intervals:
        table[start:end, channel] = 1
    return table


def preprocess_data(sequences):
    """Pack nonempty event tables and their row boundaries for distance kernels."""
    if len(sequences) == 0:
        raise ValueError("At least one sequence is required.")
    tables = [get_event_table(sequence) for sequence in sequences]
    n_channels = tables[0].shape[1]
    if n_channels == 0 or any(table.shape[1] != n_channels for table in tables):
        raise ValueError("Sequences must share a nonempty event alphabet.")
    if any(len(table) == 0 for table in tables):
        raise ValueError("Each sequence must contain a positive-duration interval.")
    offsets = np.concatenate(([0], np.cumsum([len(table) for table in tables])))
    return np.concatenate(tables), offsets.astype(np.int64)


def _validate_tables(event_tables, offsets, queries):
    tables, boundaries = np.asarray(event_tables), np.asarray(offsets)
    if tables.ndim != 2 or tables.shape[1] == 0 or not np.isin(tables, (0, 1)).all():
        raise ValueError("Event tables must be a two-dimensional binary array.")
    if (
        boundaries.ndim != 1
        or len(boundaries) < 2
        or not np.issubdtype(boundaries.dtype, np.integer)
        or boundaries[0] != 0
        or boundaries[-1] != len(tables)
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("Offsets must partition the table into nonempty sequences.")
    checked = []
    shortest = np.min(np.diff(boundaries))
    for query in queries:
        query = np.asarray(query)
        if (
            query.ndim != 2
            or query.shape[1] != tables.shape[1]
            or not 1 <= len(query) <= shortest
            or not np.isin(query, (0, 1)).all()
        ):
            raise ValueError("Queries must be binary, share the alphabet and fit every sequence.")
        checked.append(np.ascontiguousarray(query, dtype=np.int32))
    return (
        np.ascontiguousarray(tables, dtype=np.int32),
        np.ascontiguousarray(boundaries, dtype=np.int64),
        checked,
    )


@njit(cache=True, parallel=True)
def _distance_kernel(tables, offsets, query, pruning):
    """Exact mismatch minima; pruning 0=none, 1=bounds, 2=early abandon."""
    duration, n_channels = query.shape
    query_counts = np.zeros(n_channels, dtype=np.int64)
    for t in range(duration):
        for c in range(n_channels):
            query_counts[c] += query[t, c]
    query_total = np.sum(query_counts)
    distances = np.empty(len(offsets) - 1, dtype=np.float64)
    for example in prange(len(offsets) - 1):
        start, end = offsets[example], offsets[example + 1]
        best = duration * n_channels + 1
        # Mutable scratch arrays belong to this example, never to a thread peer.
        counts = np.zeros(n_channels, dtype=np.int64)
        order = np.arange(n_channels)
        for window in range(start, end - duration + 1):
            if pruning:
                if window == start:
                    for t in range(duration):
                        for c in range(n_channels):
                            counts[c] += tables[window + t, c]
                else:
                    for c in range(n_channels):
                        counts[c] += tables[window + duration - 1, c] - tables[window - 1, c]
                if abs(query_total - np.sum(counts)) >= best:
                    continue
                difference = np.abs(query_counts - counts)
                if np.sum(difference) >= best:
                    continue
                if pruning == 2:
                    order = np.argsort(-difference)
            mismatch = 0
            for position in range(n_channels):
                c = order[position]
                for t in range(duration):
                    mismatch += abs(tables[window + t, c] - query[t, c])
                if pruning == 2 and mismatch >= best:
                    break
            if mismatch < best:
                best = mismatch
            if best == 0:
                break
        distances[example] = best
    return distances


def features_from_tables(
    event_tables, offsets, queries, *, method="abide", distance="squared", validate=True
):
    """Return an (n_sequences, n_queries) matrix of sliding-window minima.

    validate=False is only for validated tables and decoder-generated queries,
    avoiding a full dataset scan in each fitness evaluation.
    """
    methods = {"reference": 0, "abide": 1, "full": 2}
    if method not in methods or distance not in ("squared", "euclidean"):
        raise ValueError("Unknown distance method or distance scale.")
    queries = list(queries)
    if not queries:
        raise ValueError("At least one query is required.")
    if validate:
        event_tables, offsets, queries = _validate_tables(event_tables, offsets, queries)
    features = np.column_stack(
        [_distance_kernel(event_tables, offsets, query, methods[method]) for query in queries]
    )
    return np.sqrt(features) if distance == "euclidean" else features


def abide_reference(event_tables, offsets, query):
    """Unpruned reference search, returning one mismatch count per sequence."""
    return features_from_tables(event_tables, offsets, [query], method="reference")[:, 0]


def abide(event_tables, offsets, query):
    """Lower-bound pruning, returning one exact mismatch count per sequence."""
    return features_from_tables(event_tables, offsets, [query])[:, 0]


def abide_full(event_tables, offsets, query):
    """Lower-bound pruning plus channel-ordered early abandonment."""
    return features_from_tables(event_tables, offsets, [query], method="full")[:, 0]


def transform_sequences(sequences, queries, *, method="abide", distance="squared"):
    """Embed interval sequences using previously learned queries."""
    tables, offsets = preprocess_data(sequences)
    return features_from_tables(tables, offsets, queries, method=method, distance=distance)

"""Optional conversion to COSTI-style signed interval endpoints.

This utility is independent of Sparse-let's binary event-table pipeline.
"""

import numpy as np


def transform_to_input_format(x):
    """Convert sequences of (start, end, channel, value) tuples to flat arrays.

    Channels are nonnegative, zero-based integers; timestamps may be fractional.
    Each noninstantaneous interval emits (start, +value) and (end, -value).
    Endpoint order follows the input; no global time sorting is applied.
    Return float32 timestamps, int32 channels, float32 values, int32 offsets.
    """
    timestamps, channels, values, offsets = [], [], [], [0]
    for example in x:
        for start, end, channel, value in example:
            if not np.isfinite([start, end, value]).all() or not 0 <= start <= end:
                raise ValueError("Expected finite values and 0 <= start <= end.")
            if (
                not isinstance(channel, (int, np.integer))
                or channel < 0
                or channel > np.iinfo(np.int32).max
            ):
                raise ValueError("Channels must be nonnegative int32 indices.")
            if start == end:
                continue
            timestamps.extend((start, end))
            channels.extend((channel, channel))
            values.extend((value, -value))
        offsets.append(len(timestamps))
    if len(timestamps) > np.iinfo(np.int32).max:
        raise ValueError("Too many endpoints for int32 offsets.")
    return (
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(channels, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
        np.asarray(offsets, dtype=np.int32),
    )

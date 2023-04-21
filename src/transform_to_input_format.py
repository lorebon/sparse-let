import numpy as np


def transform_to_input_format(x):
    '''Transform list of sequences to costi input data.
    Each sequence is a list of events. Event is a tuple
    (start [float], end [float], channel [int], value [float]).
    '''
    timestamps = []
    channels = []
    values = []
    examples_s = [0]

    for example in x:
        # discard instantenous events
        not_instant = [(s, e, c, v) for (s, e, c, v) in example if s != e]
        for start, end, channel, value in not_instant:
            timestamps.append(start)
            timestamps.append(end)
            channels.append(channel)
            channels.append(channel)
            values.append(value)
            values.append(-value)
        examples_s.append(examples_s[-1]+2*len(not_instant))

    timestamps = np.asarray(timestamps).astype(np.float32)
    channels = np.asarray(channels).astype(np.int32)
    values = np.asarray(values).astype(np.float32)
    examples_s = np.asarray(examples_s).astype(np.int32)

    return timestamps, channels, values, examples_s

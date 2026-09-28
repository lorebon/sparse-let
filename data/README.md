# Datasets

[Overview](../README.md)

Six benchmark datasets are included without modification:

| Dataset | Sequences | Classes | Event labels | Maximum end timestamp | Source reference |
| --- | ---: | ---: | ---: | ---: | --- |
| AUSLAN2 | 200 | 10 | 12 | 30 | [10] |
| BLOCKS | 210 | 8 | 8 | 123 | [29] |
| CONTEXT | 240 | 5 | 54 | 284 | [30] |
| HEPATITIS | 498 | 2 | 63 | 7555 | [6] |
| PIONEER | 160 | 3 | 92 | 80 | [31] |
| SKATING | 530 | 6 | 41 | 6829 | [32] |

Source references above refer to Table 1 and the bibliography of the
[paper](https://doi.org/10.1007/978-3-031-62922-8_1). The bundled files do not
include separate dataset license metadata; retain their
original attribution when redistributing them.

## File format

Each dataset directory contains:

- `data.txt`: whitespace-separated `sequence_id event_id start end [value]` rows.
- `classes.txt`: one integer class label per sequence, ordered by sequence ID.

Sequence IDs are contiguous and zero-based; event IDs are one-based. For example:

```text
0 1 0 4
0 3 2 6
1 2 1 5
```

A matching `classes.txt` might contain:

```text
0
1
```

These files describe two sequences sharing a three-event alphabet, with class
labels 0 and 1 respectively. Missing event labels become inactive channels.
The optional value is preserved by the loader but ignored
by binary event tables. Intervals use **[start, end)**: start is included, end is
excluded. Timestamps must be nonnegative integers for the learning pipeline.
Instantaneous events are ignored; a sequence with no positive-duration interval is
rejected. Overlapping and adjacent intervals of the same label are combined.

The table begins at time zero and ends at the largest noninstantaneous interval
end. Leading inactivity is preserved. There is no resampling, interpolation,
normalization of the time origin, or automatic truncation of fractional timestamps.

## Custom datasets

Run from the repository root. For custom data, create
`my-data/EXPERIMENT/{data,classes}.txt` and run:

```bash
python src/run_experiment.py --data-dir my-data --dataset EXPERIMENT --folds 5
```

Directory names are normalized to uppercase. Every class needs at least as many
examples as folds. The default bundled-data path is relative to the source files,
so invoking a script by its full path works from another working directory.

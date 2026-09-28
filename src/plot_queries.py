"""Plot learned sparse-lets from a saved fold archive without loading Python objects."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_queries(queries, output, *, columns=2):
    """Draw active intervals with original one-based event IDs; save a PNG/PDF/SVG."""
    queries = np.asarray(queries)
    if queries.ndim != 3 or min(queries.shape) == 0 or not np.isin(queries, (0, 1)).all():
        raise ValueError("Expected a nonempty binary (queries, duration, channels) array.")
    if not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Choose a .png, .pdf or .svg output file.")
    columns = min(columns, len(queries))
    rows = (len(queries) + columns - 1) // columns
    max_active = max(int(np.any(query, axis=0).sum()) for query in queries)
    height = max(3, 0.22 * max_active + 1.5)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, height * rows), squeeze=False, constrained_layout=True
    )
    for index, ax in enumerate(axes.flat):
        if index >= len(queries):
            ax.set_visible(False)
            continue
        query = queries[index]
        active_channels = np.flatnonzero(np.any(query, axis=0))
        for position, channel in enumerate(active_channels):
            changes = np.diff(np.r_[0, query[:, channel], 0])
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            ax.broken_barh(
                list(zip(starts, ends - starts)),
                (position - 0.3, 0.6),
                facecolors=plt.get_cmap("tab20")(int(channel) % 20),
            )
        ax.set_yticks(
            range(len(active_channels)), [f"e{channel + 1}" for channel in active_channels]
        )
        ax.set_ylim(max(len(active_channels) - 0.3, 0.7), -0.7)
        ax.set_xlim(0, len(query))
        ax.set_xlabel("Time (unit intervals)")
        ax.set_ylabel("Event label")
        ax.set_title(f"Sparse-let {index + 1} | {len(active_channels)} active events")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        if not len(active_channels):
            ax.text(0.5, 0.5, "No active events", transform=ax.transAxes, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Saved fold_XX.npz")
    parser.add_argument("--output", type=Path, required=True, help="New PNG/PDF/SVG file")
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        with np.load(args.archive, allow_pickle=False) as arrays:
            output = plot_queries(arrays["queries"], args.output, columns=args.columns)
        print(output.resolve())
    except (KeyError, ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

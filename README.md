# Sparse-let

Research code for **Learning Sparse-Lets for Interpretable Classification of
Event-interval Sequences**, by Lorenzo Bonasera, Davide Duma, and Stefano Gualandi
(MIC 2024, LNCS 14754, pp. 3–18).
[Read the paper](https://doi.org/10.1007/978-3-031-62922-8_1).

SPARSE learns a small set of temporal patterns, called **sparse-lets**, for
classifying event-interval sequences. A biased random-key genetic algorithm
(BRKGA) optimizes the patterns. Each sequence is represented by its minimum
sliding-window distance to each pattern, then classified using an SVM or random
forest.

[Datasets](data/README.md) · [Python example](examples/fit_and_predict.py)

## Quick start

Use **Python 3.12**. The dependency versions in `requirements.txt` were tested
together on Windows with Python 3.12.

Create and activate a virtual environment from the repository root:

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, use `.\.venv\Scripts\python.exe` in place of
`python`; activation is optional.

Run a small experiment to verify the installation:

```bash
python src/run_experiment.py --dataset AUSLAN2 --folds 2 --generations 2 --population-size 12 --queries 2 --output results/quickstart
```

This short run evaluates two small populations per fold. The first distance
calculation compiles with Numba and can be noticeably slower. Subsequent runs
use its on-disk cache.

Results are written to `results/quickstart/`: `summary.json` contains scores and
settings, while the fold files contain learned patterns, features, and predictions.
Use a new output directory for each run, or omit `--output` to create one automatically.

To learn patterns and predict from Python, run:

```bash
python examples/fit_and_predict.py
```

## Repository guide

| Path | Contents |
| --- | --- |
| [src/run_experiment.py](src/run_experiment.py) | Experiment CLI and cross-validation |
| [src/sparselets.py](src/sparselets.py) | Chromosome decoding, fitness, and learning |
| [src/abide.py](src/abide.py) | Event tables and sliding-window distances |
| [src/load_data.py](src/load_data.py), [src/train_test_split.py](src/train_test_split.py) | Dataset loading and stratified splits |
| [data/](data/README.md) | Six benchmark datasets, format, and sources |
| [examples/fit_and_predict.py](examples/fit_and_predict.py) | A complete training and prediction example |
| [tests/](tests/) | Numerical and experiment regression tests |

## Citation and license

```bibtex
@inproceedings{bonasera2024sparselets,
  author = {Bonasera, Lorenzo and Duma, Davide and Gualandi, Stefano},
  title = {Learning Sparse-Lets for Interpretable Classification of Event-interval Sequences},
  booktitle = {MIC 2024},
  series = {Lecture Notes in Computer Science},
  volume = {14754},
  pages = {3--18},
  year = {2024},
  doi = {10.1007/978-3-031-62922-8_1}
}
```

Source code is distributed under the [MIT license](LICENSE).

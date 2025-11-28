# GeneticAdditiveModels

This repository demonstrates using Genetic Algorithms to discover the
structure of Generalized Additive Models (GAMs). It contains scripts to run
baseline and improved GA experiments, helpers for analyses, and utilities for
data and plotting.

**Which files to use**
- **`main.py`**: Baseline GA experiment. Run the baseline GA and inspect the
  results written to `outputs/`.
- **`enhanced.py`**: Improved GA implementation (entry: `improved_main`). Use
  for advanced experiments with improved initialization, mutation and
  fitness evaluation.
- **`baselines.py`**: Baseline models and comparison utilities (e.g. decision
  tree, PyGAM baseline) to compare against GA-discovered GAMs.
- **`NSGA.py`**: Utilities for multi-objective GA experiments (if used).
- **`confidence_interval.py`**: Helpers to compute confidence intervals used in
  result reporting.
- **`estimate_dof.py`**: Tools for degrees-of-freedom estimation of components.
- **`dataset_test.py`**: Quick checks and examples for dataset loading and
  preprocessing.
- **`utils/`**: Utility modules:
  - `utils/dataset.py`: dataset loaders and preprocessing helpers.
  - `utils/repro.py`: reproducible seeding helper (`set_global_seed`).
  - `utils/visualization.py`: plotting helpers for effect plots and figures.

**Outputs**
- All experiment outputs (per-seed markdown summaries, result tables and PNG
  plots) are stored in `outputs/`.

**Quick start**
1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the baseline GA:

```bash
python main.py
```

3. Run the improved GA:

```bash
python -m enhanced
```

4. Run with multiple seeds (comma-separated) using the `SEEDS` environment
   variable:

```bash
SEEDS=42,7,123 python main.py
SEEDS=42,7,123 python -m enhanced
```

**Notes & tips**
- Use `utils/repro.set_global_seed(seed)` to seed randomness for reproducibility.
- To reduce runtime for quick tests, lower `population_size` and
  `n_generations` (or `NGEN`) in `main.py` / `enhanced.py`.
- To add a new dataset, extend `utils/dataset.py` and update the dataset call
  sites in `main.py` / `enhanced.py`.

If you want, I can add a short example command that runs a quick trial
(reduced population/generations) and saves the output to `outputs/`.
# GeneticAdditiveModels

This repository explores using a Genetic Algorithm (GA) to discover the structure
of a Generalized Additive Model (GAM) for regression (the California Housing
dataset). The GA searches per-feature components (none / linear / spline)
and their hyperparameters (e.g. number of spline knots, smoothing lambda) to
build interpretable, additive models.

**High level:**
- `main.py`: baseline GA implementation that evolves simple GAM chromosomes.
- `enhanced.py`: improved GA and chromosome encoding with better initialization,
	adaptive mutation and more robust fitness evaluation (cross-validation,
	complexity penalties).
- `baselines.py`: baseline models and utilities used for comparison (decision
	tree, standard PyGAM baseline).
- `estimate_dof.py`: helper script to estimate degrees-of-freedom behavior.
- `dataset_test.py`: small dataset / data-loading checks and examples.
- `utils/`: helper modules (`dataset.py`, `repro.py`, `visualization.py`).

Installation
------------
Create a virtual environment and install dependencies listed in
`requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate  # zsh / bash
pip install -r requirements.txt
```

Quick start — running the experiments
------------------------------------
- Run the baseline GA (example seeds are set at the bottom of `main.py`):

```bash
python main.py
```

- Run the improved GA (the function `improved_main` in `enhanced.py`):

```bash
python -m enhanced
```

Note: both scripts accept a `SEEDS` environment variable (comma-separated
integers) to run multiple seeds in one invocation, for example:

```bash
SEEDS=42,7,123 python main.py
SEEDS=42,7,123 python -m enhanced
```

Files and outputs
-----------------
- Results and per-seed markdown are written to the `outputs/` directory
	(e.g. `outputs/results_seed42.md`, `outputs/improved_results_all_seeds.md`).
- Visualizations are saved as PNGs in `outputs/` (feature effect plots).

Reproducibility
---------------
- Call `set_global_seed(seed)` from `utils/repro.py` is used by the scripts to
	seed Python and NumPy RNGs. Absolute bit-for-bit reproducibility is not
	guaranteed across platforms or library versions but seeds reduce run-to-run
	variance.

Tips & next steps
-----------------
- If runs take too long, reduce the GA population / generations passed into
	`main()` / `improved_main()` (see `population_size`, `NGEN` / `n_generations`).
- Inspect per-seed markdown in `outputs/` to compare GA-GAM vs baselines.
- If you want to try a different dataset, extend `utils/dataset.py` and pass
	the appropriate splits to the GA builder functions.

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

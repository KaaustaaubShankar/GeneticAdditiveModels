# GeneticAdditiveModels

This repository demonstrates the use of Genetic Algorithms (GAs) to discover the structure of Generalized Additive Models (GAMs). It includes scripts for running experiments, utilities for data preprocessing, and tools for visualization and analysis.

## Features
- **Confidence Interval-based NSGA**: A genetic algorithm implementation that uses confidence intervals to penalize solutions, improving robustness and interpretability.
- **Utilities**: Dataset loaders, reproducibility helpers, and visualization tools.

## Installation
1. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Quick Start
### Running the Experiment
Run the confidence interval-based NSGA:

```bash
python -m confidence_interval
```

### Outputs
- Results and summaries are saved in the `outputs/` directory (e.g., `outputs/improved_results_seed42.md`, `outputs/improved_results_all_seeds.md`).
- Visualizations, such as feature effect plots, are saved as PNG files in `outputs/`.

## File Overview
- **`confidence_interval.py`**: Main script for running the confidence interval-based NSGA.
- **`utils/`**: Utility modules:
  - `dataset.py`: Dataset loaders and preprocessing helpers.
  - `repro.py`: Reproducibility helpers (e.g., `set_global_seed`).
  - `visualization.py`: Plotting helpers for visualizations.

## Tips
- **Reproducibility**: Use `set_global_seed(seed)` from `utils/repro.py` to ensure consistent results.
- **Performance**: For quicker tests, reduce `population_size` and `n_generations` in `confidence_interval.py`.
- **Extensibility**: To add a new dataset, extend `utils/dataset.py` and update the relevant dataset call sites.

## Next Steps
- Experiment with different datasets by modifying the dataset loader in `utils/dataset.py`.
- Explore the visualizations to understand the feature effects and model behavior.

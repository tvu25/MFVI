# MFVI numerical experiments

Reproducibility code for the paper

**Stability of Finite-Batch Particle Mean-Field Variational Inference
Beyond Strong Convexity**

by Vinh Nguyen and Truong Vu.

## Requirements

- Python >= 3.10
- NumPy >= 2.0
- Matplotlib

No external datasets are required.

## Files

- `pavi.py`: generic NumPy implementation of finite-batch PAVI
  (Algorithm 1 in the manuscript).
- `expanded_experiments.py`: generates the numerical figures and CSV
  tables reported in Section 9.
- `check_stepsize_halving.py`: performs the nested Brownian
  reference-step check used in the time-discretization experiment.
- `figures/`: figures generated for the manuscript.
- `results/`: raw CSV output from the main numerical experiments.
- `stepsize_halving_check.csv`: results of the additional
  reference-step halving check.

## Reproducing the experiments

From the repository root, run

    python expanded_experiments.py

to regenerate the figures and CSV tables, and

    python check_stepsize_halving.py

to reproduce the reference-step halving diagnostic.

The numerical experiments use fixed NumPy random seeds recorded
directly in the scripts.

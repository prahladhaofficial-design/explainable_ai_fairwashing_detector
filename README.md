# Explainable AI Fairwashing Detector

A modular framework for studying fairwashing, explanation fragility, and fairness-related gaps between models and their explanations.

## Overview

Explainability tools can be misleading when the explanation is faithful to a simplified surrogate but not to the true model behavior. In fairness-sensitive settings, this gap can make a model appear safer or fairer than it is.
This repository organizes multiple fairwashing and explanation-fooling scenarios into a reusable Python framework with experiments, metrics, tests, and saved results.

## Project Design

- Separate attacks, detection, explanation methods, models, metrics, and utilities into clear modules.
- Use experiment folders for reproducible scenario-specific runs.
- Save visual and tabular outputs so findings can be inspected without rerunning everything.
- Include validation and test files to support framework-level changes.

## Key Capabilities

- LaundryML-style surrogate search with fidelity, fairness, accuracy, beta, objective, and depth tracking.
- LIME/SHAP fooling experiments for explanation fragility.
- Off-manifold analysis for explanation behavior outside the training distribution.
- Fragile-interpretation experiments across multiple settings.
- Detector utilities for identifying explanation-model fairness gaps.
- Reusable dataset, model, metric, and helper modules.

## Repository Structure

```text
src/attacks/       Fairwashing and explanation-manipulation attacks.
src/detection/     Detector implementation.
src/explanations/  LIME, SHAP, LRP, and explanation helpers.
src/models/        Model architectures.
src/evaluation/    Metric and comparison logic.
experiments/       Runnable experiment suites.
results/           Generated CSVs and plots.
tests/             Regression tests.
CODE_MAP.md        Detailed code map.
```

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python validate.py
pytest tests
python experiments/laundryml_surrogate/run_experiment.py
python experiments/detection/run_experiment.py
```

## Results and Artifacts

- `results/laundryml_candidates.csv` stores surrogate candidates and objective metrics.
- `results/laundryml_pareto.png` visualizes trade-offs between competing objectives.
- `results/lime_shap_scaffolding_lime.png` and `results/lime_shap_scaffolding_shap.png` show explanation behavior.
- `results/fragile_interpretation_aggregate.png` and `results/fragile_interpretation_comparison.png` summarize fragile-interpretation experiments.
- `CODE_MAP.md` describes the framework components and extension points.

## Reproducibility Notes

- Install the listed dependencies before running scenario pipelines because the framework relies on PyTorch, scikit-learn, LIME, and SHAP tooling.
- Run one scenario at a time when debugging so generated metrics and plots remain easy to trace.
- Keep dataset names, sensitive-attribute assumptions, and random seeds fixed when comparing fairwashing gaps.
- Store new experiment outputs under results/ with descriptive prefixes that match the scenario.

## Detailed Workflow

- Install the package in editable mode so local source changes are reflected immediately.
- Run `validate.py` to check the framework setup.
- Run the test suite before modifying attacks, models, explanations, or metrics.
- Run individual experiment folders when studying a specific fairwashing scenario.
- Inspect the `results/` directory after each experiment to compare saved plots and tables.

## Implementation Details

- The framework separates attacks, explanations, detection, data, models, metrics, and helpers into independent modules.
- Experiment runners are kept outside the core package so research scenarios do not clutter reusable source code.
- The detection layer focuses on mismatches between model behavior, explanation fidelity, and fairness-sensitive outcomes.
- The attacks folder contains multiple strategies for stressing explanation reliability.
- The code map provides a practical guide for extending the framework.

## Data and Inputs

- Dataset utilities live under `src/data/` and should be the preferred entry point for data loading.
- Large or external datasets should be placed locally and referenced through loader functions.
- Experiment outputs are stored separately from source code under `results/`.
- The `data/README.md` explains how local datasets should be organized.

## How To Interpret the Results

- High explanation fidelity does not automatically imply fairness or robustness.
- Pareto-style plots should be read as trade-off surfaces rather than single-score conclusions.
- LIME and SHAP visual outputs are useful for comparing explanation behavior under stress.
- Detection results should be interpreted alongside the attack configuration and dataset assumptions.
- The framework is most valuable when comparing multiple explanation methods under the same setting.

## Limitations and Assumptions

- Some explainability libraries can be sensitive to dependency versions.
- Experiment runtime depends on the model, dataset, and explanation method used.
- Fairness conclusions depend on the selected sensitive attributes and metrics.
- The framework is research-oriented and should be validated carefully before real-world use.


## Review Guide

Use this checklist when reviewing the repository on GitHub:

- Start with the overview to understand the project objective before opening notebooks or source files.
- Read the repository structure section to identify the main entry point.
- Inspect generated outputs before rerunning experiments; they provide a quick view of the current project state.
- Review scripts and notebooks together when both are present, since notebooks often explain decisions while scripts preserve reusable execution paths.
- Check the limitations section before interpreting results too strongly.
- Use the possible extensions section to understand how the project can grow without changing its core purpose.

## Troubleshooting

Common issues and fixes:

- If imports fail, confirm that the virtual environment is activated and dependencies were installed from `requirements.txt`.
- If a notebook cell fails because a variable is missing, restart the kernel and run the notebook from the first cell.
- If a script cannot find a data file, run it from the repository root unless the README gives a different working directory.
- If plots are not regenerated, check that the output folder exists and that the script has write permission.
- If a package version causes errors, create a fresh environment and reinstall dependencies before debugging the source code.
- If GPU-related packages are unavailable, run CPU-compatible sections first and treat heavier deep-learning experiments separately.

## Development Notes

The repository is organized to be understandable before it is optimized. Most code favors clear structure, visible intermediate steps, and saved artifacts over highly compressed implementations. When extending the project, prefer small changes that keep the original experiment reproducible.

Recommended development practices:

- Keep new data files under `data/`, `datasets/`, or the existing local data folder for that project.
- Keep generated figures under `figures/`, `plots/`, `outputs/`, or `results/` depending on the existing convention.
- Add new experiments as separate scripts or clearly labeled notebook sections.
- Avoid overwriting existing outputs unless the experiment is intentionally being refreshed.
- Document any new dependency in `requirements.txt`.
- Prefer relative paths so the repository works after cloning on another machine.

## Project Status

This repository is prepared as a standalone GitHub project. The current version includes the core implementation, documentation, and available artifacts needed to understand the work. Future improvements can focus on stronger automation, more tests, richer visual summaries, and cleaner packaging where appropriate.

## Possible Extensions

- Add more fairness metrics and sensitive-attribute configurations.
- Support additional explanation methods through a common interface.
- Add experiment configuration files for repeatable sweeps.
- Build a report generator for comparing attacks across datasets.

## Tech Stack

Python, NumPy, Pandas, scikit-learn, PyTorch, Matplotlib, pytest, LIME/SHAP-style explainability tooling.

## Author

Prahladha, IIT Hyderabad

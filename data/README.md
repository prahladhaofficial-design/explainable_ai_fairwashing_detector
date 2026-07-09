# Data Directory

This directory is reserved for local datasets used by the fairwashing experiments. Large datasets are intentionally not committed so the repository remains lightweight and easy to clone.

Use the dataset loaders in `src/data/datasets.py` to generate or load the data expected by each experiment. If an experiment needs an external dataset, place it here and keep the same path convention used by the loader.

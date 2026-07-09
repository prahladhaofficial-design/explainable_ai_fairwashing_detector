#!/bin/bash
# Run all experiments for Fairwashing Detection Framework

echo "=========================================="
echo "Fairwashing Detection Framework"
echo "Running All Experiments"
echo "=========================================="

# Create results directory
mkdir -p results

# LaundryML surrogate scenario
echo ""
echo "Running LaundryML surrogate scenario..."
python experiments/laundryml_surrogate/run_experiment.py --dataset adult_income --n_candidates 50

# LIME/SHAP scaffolding scenario
echo ""
echo "Running LIME/SHAP scaffolding scenario..."
python experiments/lime_shap_scaffolding/run_experiment.py --dataset compas --n_decoy 2

# Off-manifold fairwashing scenario
echo ""
echo "Running Off-manifold fairwashing scenario..."
python experiments/off_manifold_fairwashing/run_experiment.py --dataset fashion_mnist --n_epochs 20

# Fragile interpretation scenario
echo ""
echo "Running Fragile interpretation scenario..."
python experiments/fragile_interpretation/run_experiment.py --dataset mnist --n_samples 5

# Detection Module
echo ""
echo "Running Detection Module..."
python experiments/detection/run_experiment.py --mode comprehensive

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in results/ directory"
echo "=========================================="

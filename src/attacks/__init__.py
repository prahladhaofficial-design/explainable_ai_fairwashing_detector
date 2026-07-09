"""Fairwashing Detection and Analysis Framework."""

from src.attacks.laundryml_surrogate import LaundryML, run_laundryml_experiment
from src.attacks.lime_shap_scaffolding import ScaffoldingAttack, BiasedClassifierFactory, run_scaffolding_experiment
from src.attacks.off_manifold_fairwashing import OffManifoldFairwasher, TangentSpaceProjector, TSPDefender, create_target_explanation
from src.attacks.fragile_interpretation import TopKAttack, MassCenterAttack, RandomSignPerturbation, compute_interpretation_metrics

# CODEBASE MAP - Where to Add/Modify Code

## For Evaluator/user of this codebase - Key Places to Customize

### 1. DATA LOADING (src/data/datasets.py)
**What to modify:** Add your own datasets
- Line ~45: `load_adult_income()` - Add custom tabular dataset loaders
- Line ~85: `load_propublica_recidivism()` - Add more fairness datasets
- Line ~140: `get_image_loaders()` - Add custom image datasets
- Line ~170: `get_tabular_loaders()` - Add custom tabular datasets

**How to add a new dataset:**
```python
def load_my_dataset(path=None):
    # Load your data
    df = pd.read_csv("my_data.csv")
    y = df['target'].values
    sensitive_attr = df['protected_attr'].values
    X = df.drop(['target', 'protected_attr'], axis=1).values
    return X, y, sensitive_attr, list(df.columns)
```

### 2. MODEL ARCHITECTURES (src/models/architectures.py)
**What to modify:** Add your own models
- Line ~15: `TabularClassifier` - Modify MLP architecture
- Line ~45: `BiasedClassifier` - Modify how bias is injected
- Line ~75: `Conv4` - Modify CNN for MNIST/Fashion-MNIST
- Line ~115: `VGG16` - Modify for CIFAR-10
- Line ~180: `get_model()` - Add new model types

**How to add a new model:**
```python
class MyModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### 3. EXPLANATION METHODS (src/explanations/methods.py)
**What to modify:** Add new explanation techniques
- Line ~25: `GradientExplainer` - Modify gradient computation
- Line ~45: `XGradExplainer` - Modify input*gradient
- Line ~65: `IntegratedGradientsExplainer` - Modify IG steps
- Line ~105: `LRPExplainer` - Implement proper LRP rules
- Line ~155: `LIMEExplainerWrapper` - Modify LIME parameters
- Line ~205: `SHAPExplainerWrapper` - Modify SHAP parameters
- Line ~245: `DeepLIFTExplainer` - Implement proper DeepLIFT
- Line ~265: `get_explainer()` - Add new explainers

**How to add a new explanation method:**
```python
class MyExplainer(ExplanationMethod):
    def explain(self, x, target_class=None):
        # Your explanation logic
        return explanation_np_array
```

### 4. LAUNDRYML SURROGATE SCENARIO (src/attacks/laundryml_surrogate.py)
**What to modify:** Customize rule list generation
- Line ~25: `RuleList` - Modify rule representation
- Line ~65: `LaundryML.__init__()` - Change fairness metrics
- Line ~85: `enumerate_rule_lists()` - Modify candidate generation
- Line ~155: `_tree_to_rule_list()` - Customize rule extraction
- Line ~190: `_compute_fairness()` - Add new fairness metrics
- Line ~210: `find_best_rationalization()` - Change selection criteria

**Key customization points:**
- Change `beta_values` to explore different fairness-accuracy trade-offs
- Modify `max_depth` and `min_samples_leaf` for different rule complexities
- Add new fairness metrics in `_compute_fairness()`

### 5. LIME/SHAP SCAFFOLDING SCENARIO (src/attacks/lime_shap_scaffolding.py)
**What to modify:** Customize decoy features and OOD detection
- Line ~25: `OODDetector` - Modify clustering approach
- Line ~65: `ScaffoldingAttack.__init__()` - Change decoy parameters
- Line ~95: `add_decoy_features()` - Modify decoy generation
- Line ~115: `fool_lime_explanation()` - Customize LIME fooling
- Line ~165: `fool_shap_explanation()` - Customize SHAP fooling
- Line ~215: `BiasedClassifierFactory` - Add new bias types

**Key customization points:**
- Change `n_decoy_features` to add more/less decoy features
- Modify `decoy_std` to control decoy feature variance
- Add correlation structure to decoy features

### 6. OFF-MANIFOLD FAIRWASHING SCENARIO (src/attacks/off_manifold_fairwashing.py)
**What to modify:** Customize fairwashing and TSP defense
- Line ~25: `TangentSpaceProjector` - Modify tangent space computation
- Line ~45: `fit()` - Change neighbor selection
- Line ~75: `_fit_class_projector()` - Modify PCA parameters
- Line ~115: `OffManifoldFairwasher.__init__()` - Change loss weights
- Line ~135: `compute_explanation()` - Add new explanation methods
- Line ~165: `train_step()` - Modify loss function
- Line ~215: `train()` - Change training hyperparameters
- Line ~275: `TSPDefender` - Modify projection approach

**Key customization points:**
- Change `alpha` to balance output fidelity vs explanation manipulation
- Modify `n_neighbors` and `n_components` for TSP
- Add different target explanation patterns

### 7. FRAGILE INTERPRETATION SCENARIO (src/attacks/fragile_interpretation.py)
**What to modify:** Customize attacks and metrics
- Line ~25: `InterpretationAttack` - Modify attack base class
- Line ~55: `compute_gradient_explanation()` - Change explanation method
- Line ~75: `compute_integrated_gradients()` - Modify IG parameters
- Line ~105: `TopKAttack.__init__()` - Change attack parameters
- Line ~125: `attack()` - Modify top-k attack algorithm
- Line ~205: `MassCenterAttack` - Modify center shift attack
- Line ~265: `RandomSignPerturbation` - Change random perturbation
- Line ~285: `InfluenceFunctionAttack` - Modify influence computation
- Line ~345: `compute_interpretation_metrics()` - Add new metrics

**Key customization points:**
- Change `epsilon` for different perturbation budgets
- Modify `n_iterations` and `step_size` for attack strength
- Add new dissimilarity functions

### 8. DETECTION MODULE (src/detection/detector.py)
**What to modify:** Add new detection methods
- Line ~25: `FairwashingDetector.__init__()` - Configure detection methods
- Line ~55: `detect()` - Main detection pipeline
- Line ~115: `_check_explanation_consistency()` - Modify consistency check
- Line ~155: `_check_prediction_fidelity()` - Modify fidelity check
- Line ~185: `_check_manifold_distance()` - Modify distance metric
- Line ~215: `_check_perturbation_robustness()` - Modify robustness test
- Line ~265: `_check_cross_explanation_agreement()` - Modify agreement check
- Line ~315: `RuleListFairwashingDetector` - Modify rule list detection
- Line ~365: `LIMESHAPFoolingDetector` - Modify LIME/SHAP detection

**How to add a new detection method:**
```python
def _check_my_detection(self, model1, model2, X):
    # Your detection logic
    score = compute_my_score(model1, model2, X)
    return {"score": score, "detail": ...}
```

Then add to `detect()` method and `self.methods` list.

### 9. CONFIGURATION (src/config.py)
**What to modify:** Change all hyperparameters
- Line ~15: `LAUNDRYML_CONFIG` - Modify LaundryML settings
- Line ~30: `LIME_SHAP_SCAFFOLDING_CONFIG` - Modify scaffolding settings
- Line ~55: `OFF_MANIFOLD_CONFIG` - Modify off-manifold settings
- Line ~80: `FRAGILE_INTERPRETATION_CONFIG` - Modify fragile interpretation settings
- Line ~105: `DETECTION_CONFIG` - Modify detection thresholds
- Line ~130: `MODEL_CONFIGS` - Modify model architectures

### 10. EVALUATION (src/evaluation/metrics.py)
**What to modify:** Add new evaluation metrics and plots
- Line ~25: `evaluate_laundryml()` - Modify LaundryML scenario evaluation
- Line ~55: `evaluate_lime_shap_scaffolding()` - Modify LIME/SHAP scaffolding scenario evaluation
- Line ~75: `evaluate_off_manifold()` - Modify Off-manifold scenario evaluation
- Line ~95: `evaluate_fragile_interpretation()` - Modify Fragile-interpretation scenario evaluation
- Line ~115: `evaluate_detection()` - Modify detection evaluation
- Line ~135: `_plot_pareto()` - Modify Pareto plot
- Line ~155: `_plot_training_curves()` - Modify training plots
- Line ~175: `_plot_explanation_comparison()` - Modify comparison plots

### 11. MAIN PIPELINE (src/main_pipeline.py)
**What to modify:** Orchestrate experiments
- Line ~35: `run_laundryml_pipeline()` - Modify LaundryML scenario pipeline
- Line ~105: `run_lime_shap_scaffolding_pipeline()` - Modify LIME/SHAP scaffolding scenario pipeline
- Line ~165: `run_off_manifold_pipeline()` - Modify Off-manifold scenario pipeline
- Line ~235: `run_fragile_interpretation_pipeline()` - Modify Fragile-interpretation scenario pipeline
- Line ~305: `run_detection_experiment()` - Modify detection pipeline
- Line ~355: `main()` - Modify CLI arguments

## Quick Reference: Common Modifications

### Adding a new dataset
1. Add loader function in `src/data/datasets.py`
2. Add to `loaders` dict in `get_tabular_loaders()`
3. Update `LAUNDRYML_CONFIG` or `LIME_SHAP_SCAFFOLDING_CONFIG` in `src/config.py`

### Adding a new explanation method
1. Create class in `src/explanations/methods.py`
2. Add to `get_explainer()` factory function
3. Update `explanation_methods` lists in configs

### Adding a new attack
1. Create an attack class in the appropriate `src/attacks/` module
2. Add evaluation in `src/evaluation/metrics.py`
3. Add experiment script in `experiments/`

### Adding a new detection method
1. Add method to `FairwashingDetector` in `src/detection/detector.py`
2. Add weight and threshold in `DETECTION_CONFIG`
3. Update `detect()` method to call new check

## Testing Your Changes

```bash
# Run unit tests
python -m pytest tests/test_framework.py -v

# Run a specific scenario experiment
python experiments/laundryml_surrogate/run_experiment.py --dataset adult_income

# Run detection on your scenario
python experiments/detection/run_experiment.py --mode comprehensive
```

## Important Notes

1. **GPU Usage**: Set `DEVICE = "cuda"` in `src/config.py` for GPU training
2. **Data Paths**: Modify data root paths in experiment scripts
3. **Model Checkpoints**: Save/load models using `save_checkpoint()`/`load_checkpoint()`
4. **Random Seeds**: Call `set_seed()` before each experiment for reproducibility
5. **Batch Sizes**: Adjust based on your GPU memory
6. **Explanation Methods**: Some require additional packages (LIME, SHAP)

## Contact & Support

For issues or modules about specific implementations, refer to:
- LaundryML scenario: https://github.com/aivodji/LaundryML
- LIME/SHAP scaffolding scenario: https://github.com/dylan-slack/Fooling-LIME-SHAP
- Off-manifold scenario: https://github.com/fairwashing/fairwashing
- Fragile-interpretation scenario: https://github.com/amiratag/InterpretationFragility

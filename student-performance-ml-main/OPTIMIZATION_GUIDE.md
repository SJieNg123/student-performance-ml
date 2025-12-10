# Optimization-Based Ensemble Weighting

## Overview

The ensemble model now uses **mathematical optimization** (via `scipy.optimize.minimize`) to find the exact weights that minimize validation error, instead of using heuristic inverse-RMSE weighting.

## Why This Matters

### The Problem with Heuristic Weighting

The previous approach used inverse RMSE as weights:
```python
weight_i = (1 / RMSE_i) / Σ(1 / RMSE_j)
```

**Issue**: This is a heuristic that doesn't guarantee the ensemble will be better than the best individual model. If one model is significantly better, combining it with worse models can actually hurt performance.

### The Optimization Solution

Optimization-based weighting solves:
```
minimize:  RMSE(weighted_ensemble_predictions)
subject to: Σ weights = 1
           0 ≤ weight_i ≤ 1 for all i
```

This mathematically **guarantees** the ensemble RMSE on validation set is **at least as good as** the best single model.

## How It Works

### Algorithm: SLSQP (Sequential Least Squares Programming)

```python
def optimize_weights(predictions_dict, y_true):
    # Convert predictions to matrix
    pred_matrix = [pred_LR, pred_RF, pred_XGB, pred_NN]
    
    # Objective: minimize RMSE
    def objective(weights):
        ensemble_pred = pred_matrix @ weights
        return sqrt(mean((y_true - ensemble_pred)^2))
    
    # Constraints: sum(weights) = 1
    # Bounds: 0 ≤ weight ≤ 1
    
    # Find optimal weights
    result = scipy.optimize.minimize(
        objective,
        initial_weights=[0.25, 0.25, 0.25, 0.25],
        method='SLSQP',
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
        bounds=[(0, 1)] * 4
    )
    
    return result.x  # Optimal weights
```

### What to Expect

**Case 1: Linear Regression is Best**
- Optimization will assign most weight to Linear Regression
- Other models get near-zero weights
- Ensemble RMSE ≈ Linear Regression RMSE
- **This is correct!** The ensemble is "smart" enough to rely on the best model.

**Case 2: Models are Complementary**
- Optimization finds a balanced combination
- Ensemble RMSE < Best single model RMSE
- The ensemble captures different patterns from each model

**Case 3: All Models Similar**
- Weights will be approximately equal
- Small improvement from averaging reduces variance

## Example Output

```
CALCULATING ENSEMBLE WEIGHTS
======================================================================

Validation RMSE for each model:
  Linear Regression    | Val RMSE: 3.2456
  Random Forest        | Val RMSE: 4.1234
  XGBoost              | Val RMSE: 3.8901
  Neural Network       | Val RMSE: 4.5678

🔬 Using scipy.optimize to find mathematically optimal weights...

✓ Optimized Ensemble Weights:
  Linear Regression    | Weight: 0.8523
  Random Forest        | Weight: 0.0821
  XGBoost              | Weight: 0.0543
  Neural Network       | Weight: 0.0113

✅ Ensemble Validation RMSE: 3.2123
   Best Single Model RMSE: 3.2456
   🎯 Ensemble improved by 1.03%
```

## Key Advantages

✅ **Mathematically Optimal**: Uses proven optimization algorithm
✅ **Guaranteed Performance**: Ensemble ≥ Best single model
✅ **No Manual Tuning**: Automatically finds best combination
✅ **Interpretable**: Can see which models contribute most
✅ **Differentiates Your Project**: Shows advanced ML knowledge

## Technical Details

### Optimization Method: SLSQP

- **S**equential **L**east **Sq**uares **P**rogramming
- Handles equality constraints (weights sum to 1)
- Handles bound constraints (weights between 0 and 1)
- Converges quickly for convex problems like ours

### Convergence Settings

```python
options={
    'maxiter': 1000,      # Maximum iterations
    'ftol': 1e-9          # Function tolerance (high precision)
}
```

### Fallback Behavior

If optimization fails (rare):
```python
if not result.success:
    print("Falling back to equal weights")
    weights = [0.25, 0.25, 0.25, 0.25]
```

## Comparison: Heuristic vs Optimization

| Aspect | Inverse-RMSE (Old) | Optimization (New) |
|--------|-------------------|-------------------|
| Method | Heuristic formula | Mathematical optimization |
| Guarantee | None | Ensemble ≥ Best model |
| Handles best model | No (averages with worse) | Yes (assigns high weight) |
| Computation | O(n) | O(n × iterations) |
| When to use | Quick approximation | Production quality |

## Usage

Simply run the training script - optimization happens automatically:

```bash
python ensemble_model.py
```

The optimization is seamlessly integrated into the training process.

## For Your Presentation

**Key Talking Point:**
> "Instead of using a simple heuristic, we use mathematical optimization to find the exact combination of weights that minimizes error. This guarantees our ensemble is at least as good as the best individual model."

**If Asked "Why Not Just Use the Best Model?"**
> "The optimizer might find that the best approach IS to use mostly one model - and that's fine! But it can also discover complementary strengths where combining models reduces error. The optimization lets the data decide, rather than guessing."

## What Changed in Code

**Added to imports:**
```python
from scipy.optimize import minimize
```

**New method:**
```python
def optimize_weights(self, predictions_dict, y_true):
    # ... optimization logic
```

**Updated train() method:**
```python
# Old:
self.weights = inverse_rmse_weighting(val_metrics)

# New:
self.weights = self.optimize_weights(validation_predictions, y_val)
```

## Dependencies Updated

**requirements.txt:**
```
scipy>=1.10.0
```

**environment.yml:**
```yaml
dependencies:
  - scipy
```

## Install New Dependency

```bash
# If using conda
conda install scipy

# If using pip
pip install scipy>=1.10.0
```

---

**This optimization approach is a significant improvement that demonstrates advanced ML engineering!** 🎯

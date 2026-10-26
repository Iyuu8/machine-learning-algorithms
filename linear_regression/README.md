# Linear Regression

## Intuition
Fit a linear relationship by minimizing mean squared error with gradient descent.

## Core update
`w := w - α * ∇MSE(w)` with bias handled as an appended constant feature.

## Files
- `model.py`: gradient descent model.
- `main.py`: preprocessing, training, and test MAE output.

## From-scratch vs libraries
- From-scratch optimization and prediction.
- Pandas/sklearn used for data prep/split.

## Dependencies
- `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`

## Usage
```bash
python linear_regression/main.py
```

## Dataset
- `linear_regression/student_performance_dataset.csv` (committed).

## Limitations / caveats
- Reported "cost reduction" is optimization progress, not predictive accuracy.
- Standardization can divide by zero if a feature has no variance.

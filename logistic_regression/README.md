# Logistic Regression

## Intuition
Model class probability with sigmoid on linear logits and optimize log-loss via gradient descent.

## Core update
`p = sigmoid(Xw)`, `w := w - α * (1/m) Xᵀ(p - y)`.

## Files
- `model.py`: logistic regression training/testing and confusion matrix helpers.
- `main.py`: feature engineering, training, evaluation, feature-weight plot.

## From-scratch vs libraries
- From-scratch logistic regression optimization.
- Pandas/sklearn/matplotlib for preprocessing, split, and visualization.

## Dependencies
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Usage
```bash
python logistic_regression/main.py
```

## Dataset
- `logistic_regression/dataset/credit_customers.csv` (committed).

## Limitations / caveats
- Classification threshold is intentionally `0.80` (not default 0.50).
- Sigmoid input is clipped for overflow stability.
- Standardization can divide by zero on constant columns.

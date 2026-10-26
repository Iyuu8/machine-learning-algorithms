# Decision Tree Classifier

## Intuition
Recursively split data to reduce impurity (Gini), creating leaf predictions from majority class.

## Core math
Gini impurity: `1 - Σ p(c)^2`; choose split maximizing impurity reduction.

## Files
- `model.py`: tree/leaf structures and split logic.
- `main.py`: Iris data prep + evaluation.

## From-scratch vs libraries
- From-scratch decision tree implementation.
- `train_test_split` used for partitioning.

## Dependencies
- `numpy`, `pandas`, `scikit-learn`

## Usage
```bash
python decision_tree/main.py
```

## Dataset
- `decision_tree/dataset/Iris.csv` (committed).

## Limitations / caveats
- No pruning or feature-importance reporting.
- Standardization does not guard zero-variance features.

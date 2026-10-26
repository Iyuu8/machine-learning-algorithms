# Decision Tree Classifier

## Intuition
Recursively split data to reduce impurity (Gini), creating leaf predictions from majority class.

## Core math
Gini impurity: `1 - Σ p(c)^2`; choose split maximizing impurity reduction.

## Theory from First Principles
### 1) Goal
Learn human-readable if/else rules that partition feature space into mostly pure class regions.

### 2) Approach
Starting at the root, the model scans candidate thresholds from unique feature values. For each split it computes weighted child impurity and chooses the split with maximum Gini gain. Recursion stops at depth/min-split/min-leaf limits or pure nodes; leaves predict majority class.

### 3) Result / Output
A binary tree of decision nodes (`feature`, `threshold`) and leaf predictions. Inference routes each sample down the tree until a leaf class is returned.

### 4) Formulas
- Node impurity: `G(S)=1-\sum_c p(c\mid S)^2`
- Weighted post-split impurity: `G_{split}=\frac{|S_L|}{|S|}G(S_L)+\frac{|S_R|}{|S|}G(S_R)`
- Gain: `\Delta G = G(S)-G_{split}`

Variables: `S` = samples at current node, `S_L,S_R` = left/right subsets after threshold test, `p(c|S)` = class proportion.

### 5) Intuition (small example)
Suppose petal length cleanly separates two Iris species. A threshold near that separation greatly lowers impurity, so the tree chooses it early, making later decisions simple refinements.


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

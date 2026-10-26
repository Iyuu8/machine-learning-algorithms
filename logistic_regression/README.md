# Logistic Regression

## Intuition
Model class probability with sigmoid on linear logits and optimize log-loss via gradient descent.

## Core update
`p = sigmoid(Xw)`, `w := w - α * (1/m) Xᵀ(p - y)`.

## Theory from First Principles
### 1) Goal
Estimate class membership probability for binary classification and convert it to class labels.

### 2) Approach
A linear score `z=Xw` is passed through a sigmoid to produce probabilities. The model minimizes logistic cross-entropy with gradient descent, clipping probabilities for numerical stability. In this implementation, final class prediction uses threshold `0.80` (not `0.50`).

### 3) Result / Output
Outputs include predicted probabilities, binary labels, accuracy, and confusion-matrix statistics (`tp`, `tn`, `fp`, `fn`, precision, recall, F1).

### 4) Formulas
- Logit: `z_i = x_i^T w`
- Sigmoid: `\sigma(z)=\frac{1}{1+e^{-z}}`
- Probability: `p_i=\sigma(z_i)`
- Loss: `J(w)=-\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]`
- Gradient: `\nabla J(w)=\frac{1}{m}X^T(p-y)`
- Update: `w \leftarrow w-\alpha \nabla J(w)`
- Classification rule (repo): `\hat{y}_i = 1` if `p_i\ge 0.80`, else `0`.

### 5) Intuition (small example)
For credit-risk data, points on one side of the decision boundary get probabilities near 1 and the other side near 0. If a risky customer gets high `p`, gradient descent pushes weights to reduce that error on similar samples in future iterations.


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

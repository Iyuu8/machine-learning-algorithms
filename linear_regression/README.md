# Linear Regression

## Intuition
Fit a linear relationship by minimizing mean squared error with gradient descent.

## Core update
`w := w - α * ∇MSE(w)` with bias handled as an appended constant feature.

## Theory from First Principles
### 1) Goal
Learn a linear mapping from numeric features to a continuous target by minimizing prediction error.

### 2) Approach
The implementation augments each feature vector with a constant `1` (bias term), initializes weights, then performs gradient descent for up to 1000 iterations. At each step it computes predictions `\hat{y}=Xw`, evaluates error, and updates weights along the negative MSE gradient.

### 3) Result / Output
The model returns fitted weights and predictions. `test(...)` reports mean squared error and mean absolute error, where lower values indicate better fit.

### 4) Formulas
- Design matrix with bias: `X_b = [X\ \mathbf{1}]`
- Prediction: `\hat{y} = X_b w`
- Objective: `J(w)=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2`
- Gradient: `\nabla J(w)=\frac{2}{m}X_b^T(X_b w-y)`
- Update: `w \leftarrow w-\alpha \nabla J(w)`

Variables: `m` = number of samples, `d` = number of original features, `w\in\mathbb{R}^{d+1}` includes bias, `\alpha` = learning rate.

### 5) Intuition (small example)
If study hours increase exam score roughly linearly, the algorithm repeatedly adjusts slope/intercept so predicted scores move closer to observed scores. The gradient points toward over-predicted vs under-predicted regions, so updates systematically reduce average squared error.


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

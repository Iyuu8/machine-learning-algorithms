# Support Vector Machine (Linear)

## Intuition
Learn a separating hyperplane maximizing margin while penalizing hinge violations.

## Core objective (informal)
Minimize regularization + hinge-loss tradeoff over linear decision function.

## Theory from First Principles
### 1) Goal
Find a linear boundary that separates two classes with maximum margin while penalizing margin violations.

### 2) Approach
The implementation uses labels in `{−1,+1}` and optimizes a primal hinge-loss objective by gradient steps on weights and bias. Samples with margin `<1` contribute to the loss gradient; well-classified high-margin samples do not.

### 3) Result / Output
Returns learned `(w,b)` and accuracy metrics based on sign of `w^Tx+b` (via margin checks).

### 4) Formulas
- Margin for sample `i`: `m_i = y_i (w^T x_i + b)`
- Hinge loss: `\ell_i = \max(0,1-m_i)`
- Objective: `J(w,b)=\frac{1}{2}\lVert w\rVert_2^2 + C\frac{1}{m}\sum_{i=1}^{m}\ell_i`
- Subgradient-style update (matching implementation structure):
  - `dW = w - C\frac{1}{m}\sum_{i:m_i<1} y_i x_i`
  - `db = -C\frac{1}{m}\sum_{i:m_i<1} y_i`
  - `w\leftarrow w-\alpha dW`, `b\leftarrow b-\alpha db`

### 5) Intuition (small example)
Among many separating lines, SVM prefers one leaving the widest gap between classes. Points deep inside the correct side have zero hinge penalty; boundary-violating points pull the line back toward a better separator.


## Files
- `model.py`: from-scratch linear SVM-style training logic.
- `main.py`: preprocessing, custom model run, sklearn linear SVC comparison.

## From-scratch vs libraries
- From-scratch SVM-style implementation.
- Scikit-learn `SVC` used as comparison baseline.

## Dependencies
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Usage
```bash
python support_vector_machine/main.py
```

## Dataset
- `support_vector_machine/dataset/Exam_Score_Prediction.csv` (committed).

## Limitations / caveats
- Standardization can divide by zero if a feature is constant.
- Focused on educational linear setup (no kernelized custom implementation).

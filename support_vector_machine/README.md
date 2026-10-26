# Support Vector Machine (Linear)

## Intuition
Learn a separating hyperplane maximizing margin while penalizing hinge violations.

## Core objective (informal)
Minimize regularization + hinge-loss tradeoff over linear decision function.

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

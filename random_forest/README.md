# Random Forest Regression

## Intuition
Aggregate many randomized decision trees trained on bootstrap samples and average predictions.

## Core rule
`ŷ = (1/T) Σ_t tree_t(x)`.

## Files
- `model.py`: from-scratch regression tree + forest.
- `main.py`: preprocessing, training, MAE output, sklearn comparison.

## From-scratch vs libraries
- From-scratch tree and forest logic.
- Scikit-learn regressor used for reference comparison.

## Dependencies
- `numpy`, `pandas`, `scikit-learn`

## Usage
```bash
python random_forest/main.py
```

## Dataset
- `random_forest/dataset/Exam_Score_Prediction.csv` (committed).

## Limitations / caveats
- No OOB scoring or feature-importance APIs.
- Runtime increases with number/depth of trees.

# Random Forest Regression

## Intuition
Aggregate many randomized decision trees trained on bootstrap samples and average predictions.

## Core rule
`ŷ = (1/T) Σ_t tree_t(x)`.

## Theory from First Principles
### 1) Goal
Improve regression accuracy and stability by averaging many diverse high-variance decision trees.

### 2) Approach
Each tree is trained on a bootstrap sample (`bagSize` fraction with replacement). During splits, each tree evaluates only a random subset of features (about `d/3` here), reducing correlation across trees. Final prediction is the mean of tree outputs.

### 3) Result / Output
Outputs one continuous prediction per sample. Averaging reduces variance compared with a single tree, usually improving generalization.

### 4) Formulas
For tree `t` prediction `f_t(x)` and `T` trees:
- Ensemble prediction: `\hat{y}(x)=\frac{1}{T}\sum_{t=1}^{T}f_t(x)`

Inside each regression tree, split selection maximizes MSE reduction:
- `\text{MSE}(S)=\frac{1}{|S|}\sum_{i\in S}(y_i-\bar{y}_S)^2`
- `\Delta=\text{MSE}(S)-\left(\frac{|S_L|}{|S|}\text{MSE}(S_L)+\frac{|S_R|}{|S|}\text{MSE}(S_R)\right)`

### 5) Intuition (small example)
If one tree overfits noisy exam-score samples, other trees trained on different bootstraps tend to make different errors. Averaging their predictions cancels part of this noise.


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

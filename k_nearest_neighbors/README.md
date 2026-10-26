# K-Nearest Neighbors (KNN)

## Intuition
Classify a sample using the majority label among its `k` nearest training samples.

## Core rule
Compute squared distances to all training points, select `k` nearest, vote by class frequency.

## Theory from First Principles
### 1) Goal
Classify unseen samples using local similarity to labeled training data.

### 2) Approach
KNN is instance-based: training stores data only. At prediction time, the implementation computes squared Euclidean distances from each test point to all training points, selects `k` closest indices (`argpartition`), then applies majority vote using `np.bincount(...).argmax()`.

### 3) Result / Output
Produces a predicted class per test sample plus aggregate accuracy (`correct / total`).

### 4) Formulas
- Distance (squared Euclidean): `d(x,x_i)=\sum_{j=1}^{d}(x_j-x_{ij})^2`
- Neighbor set: `N_k(x)=\text{k samples with smallest }d(x,x_i)`
- Prediction: `\hat{y}=\arg\max_c\sum_{x_i\in N_k(x)}\mathbf{1}[y_i=c]`

Variables: `d` = number of features, `c` = class label, `k` = neighbors count.

### 5) Intuition (small example)
If 5 nearest flowers to a new Iris sample are labeled `[setosa,setosa,setosa,versicolor,setosa]`, majority vote predicts `setosa`. KNN works because nearby points often share labels after proper feature scaling.


## Files
- `model.py`: vectorized distance and vote prediction.
- `main.py`: Iris preprocessing and evaluation.

## From-scratch vs libraries
- From-scratch KNN prediction logic.
- `train_test_split` used for data partitioning.

## Dependencies
- `numpy`, `pandas`, `scikit-learn`

## Usage
```bash
python k_nearest_neighbors/main.py
```

## Dataset
- `k_nearest_neighbors/dataset/Iris.csv` (committed).

## Limitations / caveats
- Current implementation expects integer-encoded labels for bincount voting.
- `k` must be in `[1, number_of_training_samples]`.

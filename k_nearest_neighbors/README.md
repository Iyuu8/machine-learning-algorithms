# K-Nearest Neighbors (KNN)

## Intuition
Classify a sample using the majority label among its `k` nearest training samples.

## Core rule
Compute squared distances to all training points, select `k` nearest, vote by class frequency.

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

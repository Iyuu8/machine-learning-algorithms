# K-Means

## Intuition
Alternate between assigning each sample to its nearest centroid and updating centroids to cluster means.

## Core update
1. Assign: `c_i = argmin_j ||x_i - μ_j||^2`
2. Update: `μ_j = mean({x_i | c_i = j})`

## Files
- `model.py`: K-means++-style init, assignment, updates, silhouette score.
- `main.py`: run across `k` values and plot cost/silhouette curves.

## From-scratch vs libraries
- From-scratch clustering logic.
- Matplotlib/pandas for plotting and loading only.

## Dependencies
- `numpy`, `pandas`, `matplotlib`

## Usage
```bash
python k_means/main.py
```

## Dataset
- Requires `k_means/dataset/Country-data.csv` (not committed).

## Limitations / caveats
- Standardization can divide by zero on constant columns.
- Results depend on random initialization.

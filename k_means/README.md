# K-Means

## Intuition
Alternate between assigning each sample to its nearest centroid and updating centroids to cluster means.

## Core update
1. Assign: `c_i = argmin_j ||x_i - μ_j||^2`
2. Update: `μ_j = mean({x_i | c_i = j})`

## Theory from First Principles
### 1) Goal
Partition unlabeled samples into `k` compact groups with low within-cluster variance.

### 2) Approach
This implementation uses K-means++-style seeding: first centroid uniform random, later centroids sampled with probability proportional to squared distance from existing centroids. Then it alternates assignment and centroid update for a fixed number of epochs.

### 3) Result / Output
Returns final centroids and the clustering objective `J` (sum of squared distances to nearest centroid). It also computes silhouette score for separation/cohesion diagnostics.

### 4) Formulas
- Assignment: `c_i=\arg\min_j\lVert x_i-\mu_j\rVert_2^2`
- Update: `\mu_j=\frac{1}{|C_j|}\sum_{i:c_i=j}x_i`
- Objective: `J=\sum_{i=1}^{m}\lVert x_i-\mu_{c_i}\rVert_2^2`
- Silhouette for sample `i`: `s_i=\frac{b_i-a_i}{\max(a_i,b_i)}` where `a_i` is mean intra-cluster distance and `b_i` nearest-cluster mean distance.

### 5) Intuition (small example)
With customer income and spending features, nearest-centroid assignment forms rough segments, then centroids move to each segment center. Repeating this assign-then-recenter loop stabilizes cluster structure.


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

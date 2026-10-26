# PCA

## Intuition
Project data onto top-variance orthogonal directions (principal components).

## Core math
Compute covariance, eigendecompose, sort eigenpairs descending, keep top-`k` eigenvectors.

## Files
- `model.py`: compress/decompress PCA operations.
- `main.py`: variance/error scan over `k`, plus 3D projection visualization.

## From-scratch vs libraries
- From-scratch PCA decomposition logic.
- Scikit-learn `KMeans` used only for projection visualization grouping.

## Dependencies
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Usage
```bash
python pca/main.py
```

## Dataset
- Requires `pca/dataset/Country-data.csv` (not committed).

## Limitations / caveats
- `k` must be in `[1, number_of_features]`.
- Input should be standardized before PCA.

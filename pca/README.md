# PCA

## Intuition
Project data onto top-variance orthogonal directions (principal components).

## Core math
Compute covariance, eigendecompose, sort eigenpairs descending, keep top-`k` eigenvectors.

## Theory from First Principles
### 1) Goal
Reduce dimensionality while preserving as much variance (information) as possible.

### 2) Approach
Given centered/standardized data matrix `X`, the implementation forms covariance-like matrix `(1/n)X^TX`, computes eigenpairs, sorts them descending by eigenvalue, keeps first `k` eigenvectors in `W`, and projects data to `Z=XW`. Reconstruction uses `\hat{X}=ZW^T`.

### 3) Result / Output
`compress()` returns reduced representation, projection matrix, full eigenvalue spectrum, and retained eigenvalues. `deCompress()` returns reconstructed data and absolute reconstruction error sum.

### 4) Formulas
- Covariance matrix: `\Sigma=\frac{1}{n}X^TX`
- Eigenproblem: `\Sigma v_j = \lambda_j v_j`
- Projection matrix: `W=[v_1,\dots,v_k]` (largest `\lambda_j`)
- Compression: `Z=XW`
- Reconstruction: `\hat{X}=ZW^T`
- Explained variance ratio (concept): `\frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{d}\lambda_j}`

### 5) Intuition (small example)
If 2D points lie near a diagonal line, one principal component captures most variation. Projecting onto that direction keeps structure while dropping the low-variance orthogonal noise direction.


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

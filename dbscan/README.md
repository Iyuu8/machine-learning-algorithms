# DBSCAN

## Intuition
Cluster points by density: core points have enough neighbors in an epsilon-radius neighborhood, and clusters expand through density reachability.

## Core rule
Neighborhood: `Nε(p) = {q | distance(p, q) < ε}` (self excluded in this implementation).

## Theory from First Principles
### 1) Goal
Discover clusters of arbitrary shape by grouping dense regions and labeling sparse points as noise/outliers.

### 2) Approach
The code precomputes pairwise distances, then repeatedly starts from an unvisited point. If its `ε`-neighborhood size is at least `m`, it becomes a seed for a new cluster and a queue expands density-reachable points. This repository intentionally excludes the point itself from `N_\varepsilon`, uses `count >= m` for seed creation, and `count > m` for queue expansion.

### 3) Result / Output
Returns an integer cluster label per sample (`-1` means noise or unassigned).

### 4) Formulas
- Neighborhood: `N_\varepsilon(p)=\{q\neq p\mid \lVert q-p\rVert_2<\varepsilon\}`
- Core-point test (seed phase): `|N_\varepsilon(p)|\ge m`
- Expansion condition (this implementation): expand neighbor `q` when `|N_\varepsilon(q)|>m`

Variables: `\varepsilon` radius, `m` minimum neighbor count, `\lVert\cdot\rVert_2` Euclidean norm.

### 5) Intuition (small example)
Imagine points shaped like two moons. K-means struggles because clusters are non-spherical, but DBSCAN chains nearby dense points together, so each moon can become one cluster while isolated points remain noise.


## Files
- `model.py`: DBSCAN clustering logic.
- `generate_dataset.py`: deterministic toy 3D dataset generator.
- `main.py`: clustering run + 3D plots.
- `visualize_dataset.py`: standalone dataset visualization.

## From-scratch vs libraries
- From-scratch DBSCAN logic.
- Pandas/Matplotlib only for data IO and visualization.

## Dependencies
- `numpy`, `pandas`, `matplotlib`

## Usage
```bash
python dbscan/generate_dataset.py
python dbscan/main.py
```

## Dataset
- Generated at `dbscan/dataset/dbscan_toy_3d.csv`.

## Limitations / caveats
- Current convention excludes self from neighbor count.
- Seed check uses `count >= m`; expansion currently uses `count > m` (preserved intentionally).

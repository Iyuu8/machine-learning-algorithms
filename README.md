# Machine Learning Algorithms (Educational Implementations)

This repository contains educational algorithm implementations made for learning and portfolio demonstration. Most models are implemented from scratch with NumPy, and some scripts include scikit-learn utilities for data splitting, preprocessing, or side-by-side comparison.

## Table of Contents
- [Project Purpose](#project-purpose)
- [Repository Structure](#repository-structure)
- [Algorithm Catalogue](#algorithm-catalogue)
- [From-Scratch vs Library Usage](#from-scratch-vs-library-usage)
- [Installation](#installation)
- [How to Run Each Example](#how-to-run-each-example)
- [Datasets and External Requirements](#datasets-and-external-requirements)
- [Reproducibility Notes](#reproducibility-notes)
- [Testing and Verification](#testing-and-verification)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Repository Metadata Recommendations](#repository-metadata-recommendations)

## Project Purpose
- Portfolio repository for machine learning / AI internship applications.
- Focus: readable educational code, algorithm intuition, and practical experimentation.
- Scope includes machine learning, reinforcement learning, and classical search/AI algorithms.

## Repository Structure
- `constraint_satisfaction/`
- `dbscan/`
- `decision_tree/`
- `deep_learning_mlp/`
- `dijkstra_algorithm/`
- `hill_climbing_hospital_configuration/`
- `k_means/`
- `k_nearest_neighbors/`
- `linear_regression/`
- `logistic_regression/`
- `pca/`
- `random_forest/`
- `reinforcement_learning_Q_Sarsa/`
- `support_vector_machine/`
- `tic_tac_toe_minimax/`
- `tests/` (lightweight deterministic checks)

## Algorithm Catalogue

### Supervised Learning
- `linear_regression/`: linear regression via gradient descent.
- `logistic_regression/`: logistic regression classifier (custom threshold currently `0.80`).
- `k_nearest_neighbors/`: KNN classifier.
- `decision_tree/`: classification tree (Gini-based splitting).
- `random_forest/`: random forest regressor (bagged from-scratch trees).
- `support_vector_machine/`: linear SVM-style classifier.

### Unsupervised Learning
- `k_means/`: K-means++-style centroid initialization and clustering.
- `dbscan/`: density-based clustering with explicit epsilon neighborhood handling.
- `pca/`: principal component analysis using covariance eigendecomposition.

### Deep Learning
- `deep_learning_mlp/`: MLP from scratch for digit classification.

### Reinforcement Learning
- `reinforcement_learning_Q_Sarsa/`: Q-learning and SARSA on racetrack-style environments.

### Search / AI Algorithms
- `constraint_satisfaction/`: backtracking + heuristics for CSP assignment.
- `dijkstra_algorithm/`: shortest path with priority queue.
- `hill_climbing_hospital_configuration/`: hill climbing + random restarts.
- `tic_tac_toe_minimax/`: minimax-based Tic-Tac-Toe agent.

## From-Scratch vs Library Usage
- **From-scratch core implementations**: decision tree, KNN, linear/logistic regression, K-means, DBSCAN, PCA, MLP, random forest, SVM, Q-learning/SARSA, CSP, Dijkstra, hill climbing, minimax.
- **Library-assisted parts**: `train_test_split`, one-hot encoding, plotting, and occasional comparison with scikit-learn models (e.g., SVM / random forest / PCA visualization).

## Installation
From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run Each Example
Run from repository root unless noted.

### Supervised
- Decision tree:
  ```bash
  python decision_tree/main.py
  ```
- KNN:
  ```bash
  python k_nearest_neighbors/main.py
  ```
- Linear regression:
  ```bash
  python linear_regression/main.py
  ```
- Logistic regression:
  ```bash
  python logistic_regression/main.py
  ```
- Random forest regression:
  ```bash
  python random_forest/main.py
  ```
- Linear SVM:
  ```bash
  python support_vector_machine/main.py
  ```

### Unsupervised
- K-means:
  ```bash
  python k_means/main.py
  ```
- DBSCAN dataset generation (reproducible toy data, seed fixed in script):
  ```bash
  python dbscan/generate_dataset.py
  ```
- DBSCAN clustering and plots:
  ```bash
  python dbscan/main.py
  ```
- DBSCAN dataset visualization:
  ```bash
  python dbscan/visualize_dataset.py
  ```
- PCA:
  ```bash
  python pca/main.py
  ```

### Deep Learning
- MLP (MNIST-style Kaggle CSV expected):
  ```bash
  python deep_learning_mlp/main.py
  ```

### Reinforcement Learning
- Q-learning/SARSA race track simulation:
  ```bash
  python reinforcement_learning_Q_Sarsa/main.py
  ```

### Search / AI
- Constraint satisfaction:
  ```bash
  python constraint_satisfaction/constraints.py
  ```
- Dijkstra:
  ```bash
  python dijkstra_algorithm/main.py
  ```
- Hill climbing hospital configuration:
  ```bash
  python hill_climbing_hospital_configuration/hospitals.py
  ```
- Tic-Tac-Toe minimax:
  ```bash
  python tic_tac_toe_minimax/main.py
  ```

## Datasets and External Requirements
Already included in repository:
- `decision_tree/dataset/Iris.csv`
- `k_nearest_neighbors/dataset/Iris.csv`
- `logistic_regression/dataset/credit_customers.csv`
- `random_forest/dataset/Exam_Score_Prediction.csv`
- `support_vector_machine/dataset/Exam_Score_Prediction.csv`
- `linear_regression/student_performance_dataset.csv`

Required but currently **not committed**:
- `k_means/dataset/Country-data.csv`
- `pca/dataset/Country-data.csv`
- `deep_learning_mlp/digit-recognizer/train.csv` (Kaggle digit recognizer format)

DBSCAN dataset:
- Generated locally by `dbscan/generate_dataset.py` into `dbscan/dataset/dbscan_toy_3d.csv`.

## Reproducibility Notes
- Some scripts use random initialization without setting seeds everywhere; results may vary run-to-run.
- DBSCAN implementation currently excludes self from epsilon-neighbors and uses mixed `>= m` (seed) vs `> m` (expansion) checks; this convention is documented and preserved.
- Several scripts use feature standardization by division with per-feature standard deviation; zero-variance features can cause divide-by-zero unless pre-cleaned.

## Testing and Verification
Lightweight deterministic tests are in `tests/` and can be run with:

```bash
pytest -q
```

These tests intentionally avoid long training runs and unavailable external datasets.

## Limitations and Future Improvements
- Add standardized data-loading helpers shared across folders.
- Add consistent input validation for all algorithms.
- Add optional fast smoke scripts for every directory.
- Add CI workflow for lint/test execution.
- Review and potentially unify DBSCAN MinPts conventions.
- Revisit reinforcement learning update schedule: `alpha` argument exists but update currently uses visit-count-based adaptive learning rate.

## Repository Metadata Recommendations
Suggested repository description:
- `Educational machine learning and AI algorithms implemented from scratch in Python (with focused comparisons to scikit-learn).`

Suggested GitHub topics:
- `machine-learning`
- `algorithms`
- `python`
- `from-scratch`
- `deep-learning`
- `reinforcement-learning`
- `decision-tree`
- `kmeans`
- `dbscan`
- `pca`
- `svm`
- `minimax`

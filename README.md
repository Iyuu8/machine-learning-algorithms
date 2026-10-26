# Machine Learning Algorithms

Educational implementations of machine learning, reinforcement learning, and classical AI algorithms in Python. Most algorithms are implemented from scratch with NumPy, with selected comparisons and visualization using common scientific Python libraries.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Algorithm Catalogue](#algorithm-catalogue)
- [Installation](#installation)
- [Running the Examples](#running-the-examples)
- [Data and Dependencies](#data-and-dependencies)
- [Testing](#testing)

## Project Overview

This repository is designed for learning, experimentation, and portfolio demonstration. Each project focuses on making the underlying algorithm easy to explore through readable Python implementations and practical examples.

## Repository Structure

- `constraint_satisfaction/` — Constraint satisfaction with backtracking and heuristics
- `dbscan/` — Density-based clustering
- `decision_tree/` — Decision tree classification
- `deep_learning_mlp/` — Multilayer perceptron digit classification
- `dijkstra_algorithm/` — Shortest-path search
- `hill_climbing_hospital_configuration/` — Hill-climbing optimization
- `k_means/` — K-means clustering
- `k_nearest_neighbors/` — K-nearest neighbors classification
- `linear_regression/` — Linear regression
- `logistic_regression/` — Logistic regression classification
- `pca/` — Principal component analysis
- `random_forest/` — Random forest regression
- `reinforcement_learning_Q_Sarsa/` — Q-learning and SARSA
- `support_vector_machine/` — Support vector machine-style classification
- `tic_tac_toe_minimax/` — Minimax game-playing agent
- `tests/` — Lightweight automated checks

## Algorithm Catalogue

### Supervised Learning

- Linear regression
- Logistic regression
- K-nearest neighbors
- Decision trees
- Random forests
- Support vector machines

### Unsupervised Learning

- K-means clustering
- DBSCAN clustering
- Principal component analysis

### Deep Learning

- Multilayer perceptron for digit classification

### Reinforcement Learning

- Q-learning
- SARSA

### Search and AI

- Constraint satisfaction
- Dijkstra's algorithm
- Hill climbing
- Minimax

## Installation

From the repository root:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Running the Examples

Run commands from the repository root.

### Supervised Learning

```bash
python decision_tree/main.py
python k_nearest_neighbors/main.py
python linear_regression/main.py
python logistic_regression/main.py
python random_forest/main.py
python support_vector_machine/main.py
```

### Unsupervised Learning

```bash
python k_means/main.py
python dbscan/generate_dataset.py
python dbscan/main.py
python dbscan/visualize_dataset.py
python pca/main.py
```

### Deep Learning

```bash
python deep_learning_mlp/main.py
```

### Reinforcement Learning

```bash
python reinforcement_learning_Q_Sarsa/main.py
```

### Search and AI

```bash
python constraint_satisfaction/constraints.py
python dijkstra_algorithm/main.py
python hill_climbing_hospital_configuration/hospitals.py
python tic_tac_toe_minimax/main.py
```

## Data and Dependencies

The examples use Python and widely adopted scientific computing tools, including NumPy, pandas, Matplotlib, and scikit-learn utilities where appropriate. Several projects include sample data or generate demonstration data as part of their workflow. Dataset-specific instructions are available in the README files within the corresponding project directories.

## Testing

Run the repository's automated checks with:

```bash
pytest -q
```

The tests provide quick verification of the core educational implementations without requiring lengthy model training.

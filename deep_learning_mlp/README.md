# Deep Learning MLP

## Intuition
A feed-forward neural network learns weights via backpropagation to classify digit labels.

## Core update
Gradient descent updates each layer parameter using chain-rule derivatives from output loss.

## Files
- `model.py`: MLP architecture, forward pass, backprop, training.
- `main.py`: data loading, training loop, accuracy reporting.

## From-scratch vs libraries
- From-scratch neural network training logic.
- Pandas used to load CSV and encode labels.

## Dependencies
- `numpy`, `pandas`, `scikit-learn`

## Usage
```bash
python deep_learning_mlp/main.py
```

## Dataset
- Requires `deep_learning_mlp/digit-recognizer/train.csv` (Kaggle MNIST format), not committed.

## Limitations / caveats
- Training can be long on CPU.
- No automated checks in this repo for this full training path.

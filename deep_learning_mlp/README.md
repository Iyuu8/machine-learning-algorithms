# Deep Learning MLP

## Intuition
A feed-forward neural network learns weights via backpropagation to classify digit labels.

## Core update
Gradient descent updates each layer parameter using chain-rule derivatives from output loss.

## Theory from First Principles
### 1) Goal
Learn a non-linear mapping from pixel features to digit classes.

### 2) Approach
The network uses fully connected layers, ReLU activations in hidden layers, and softmax at output. Training is mini-batch gradient descent: forward pass computes class probabilities and cross-entropy loss, backward pass applies chain rule to get gradients for each weight/bias matrix, then updates parameters.

### 3) Result / Output
Produces class-probability vectors and predicted one-hot labels. Accuracy compares predicted argmax class against true argmax class.

### 4) Formulas
For layer `l`:
- Linear step: `Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}`
- Activation: `A^{[l]} = \text{ReLU}(Z^{[l]})` for hidden layers, `A^{[L]}=\text{softmax}(Z^{[L]})`
- Softmax: `\text{softmax}(z)_k=\frac{e^{z_k}}{\sum_j e^{z_j}}`
- Loss (mini-batch): `J=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{k,i}\log \hat{y}_{k,i}`
- Output gradient: `dZ^{[L]}=A^{[L]}-Y`
- Parameter gradients: `dW^{[l]}=\frac{1}{m}dZ^{[l]}(A^{[l-1]})^T`, `db^{[l]}=\frac{1}{m}\sum dZ^{[l]}`

### 5) Intuition (small example)
Early layers can learn stroke-like features (edges/curves); deeper layers combine them into digit concepts. If a 3 is predicted as 8, gradient signals push parameters to increase 3 evidence and suppress 8 evidence on similar patterns.


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

import numpy as np

from decision_tree.model import DecisionTreeClassifier


def test_decision_tree_basic_fit_predict():
    features = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    target = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(depth=2)
    model.buildTree(features, target)
    predictions = model.predict(features)

    assert np.array_equal(predictions, target)

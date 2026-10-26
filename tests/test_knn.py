import numpy as np
import pytest

from k_nearest_neighbors.model import Model


def test_knn_predicts_majority_label():
    train_features = np.array([[0.0], [1.0], [2.0], [3.0]])
    train_target = np.array([0, 0, 1, 1])
    test_features = np.array([[1.1], [2.9]])

    model = Model(3, train_features, train_target)
    predictions = model.predict(test_features)

    assert np.array_equal(predictions, np.array([0, 1]))


def test_knn_rejects_invalid_k():
    train_features = np.array([[0.0], [1.0]])
    train_target = np.array([0, 1])

    with pytest.raises(ValueError):
        Model(0, train_features, train_target)

    with pytest.raises(ValueError):
        Model(3, train_features, train_target)

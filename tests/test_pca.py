import numpy as np
import pytest

from pca.model import Model


def test_pca_compress_and_decompress_shape():
    dataset = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    ])

    model = Model(2, dataset)
    reduced_data, W, eigenvalues_all, eigenvalues_kept = model.compress()
    approx, _ = model.deCompress(reduced_data)

    assert reduced_data.shape == (4, 2)
    assert W.shape == (3, 2)
    assert approx.shape == dataset.shape
    assert np.all(eigenvalues_all[:-1] >= eigenvalues_all[1:])
    assert np.array_equal(eigenvalues_kept, eigenvalues_all[:2])


def test_pca_rejects_invalid_k():
    dataset = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        Model(0, dataset)

    with pytest.raises(ValueError):
        Model(3, dataset)

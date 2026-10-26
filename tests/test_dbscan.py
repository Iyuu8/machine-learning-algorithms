import numpy as np

from dbscan.model import Model


def test_dbscan_basic_clustering_with_current_minpts_convention():
    dataset = np.array([
        [0.0, 0.0],
        [0.0, 0.1],
        [0.1, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.0, 5.1],
        [5.1, 5.0],
        [5.1, 5.1],
        [10.0, 10.0],
    ])

    model = Model(epsilon=0.3, m=2, dataset=dataset)
    clusters = np.array(model.createClusters())

    assert len(set(clusters[:4])) == 1
    assert len(set(clusters[4:8])) == 1
    assert clusters[0] != clusters[4]
    assert clusters[8] == -1

from dijkstra_algorithm.main import shortest_path


def test_shortest_path_returns_expected_path_and_distance():
    graph = {
        "A": [(1, "B"), (4, "C")],
        "B": [(1, "A"), (2, "C"), (5, "D")],
        "C": [(4, "A"), (2, "B"), (1, "D")],
        "D": [(5, "B"), (1, "C")],
    }

    best_weights, path = shortest_path(graph, "A", "D")

    assert path == ["A", "B", "C", "D"]
    assert best_weights["D"] == 4


def test_shortest_path_unreachable_target_returns_empty_path():
    graph = {
        "A": [(1, "B")],
        "B": [(1, "A")],
        "C": [],
    }

    best_weights, path = shortest_path(graph, "A", "C")

    assert path == []
    assert best_weights["C"] == float("inf")

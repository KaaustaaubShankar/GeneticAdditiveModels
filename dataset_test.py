from utils.dataset import Dataset


def test_default_iris_shapes():
    """Smoke test: loading the iris dataset yields non-empty splits."""
    data = Dataset(default="iris", target="target", scale=True)
    X_train, X_test, y_train, y_test = data.get_splits()

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0
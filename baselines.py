from sklearn.tree import DecisionTreeRegressor
from utils.dataset import RegressionDataset
from sklearn.metrics import mean_squared_error
import numpy as np
try:
    from pygam import LinearGAM, s
    _HAS_PYGAM = True
except Exception:
    # pygam not available; provide lightweight fallbacks so rest of the
    # pipeline can continue. The fallback `PyGAMRegressionBaseline` below
    # will detect this and behave accordingly.
    LinearGAM = None
    s = None
    _HAS_PYGAM = False
from utils.repro import set_global_seed

# ------------------------------
# Decision Tree Regression Baseline
# ------------------------------
class DecisionTreeRegressionBaseline:
    def __init__(self, max_depth=None, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))  # RMSE

# ------------------------------
# PyGAM Regression Baseline
# ------------------------------
class PyGAMRegressionBaseline:
    def __init__(self, n_splines=25, max_iter=100):
        self.n_splines = n_splines
        self.max_iter = max_iter
        self.model = None

    def fit(self, X_train, y_train):
        if not _HAS_PYGAM:
            # fallback: store None model and skip fitting
            self.model = None
            return
        # Create GAM terms for each feature
        terms = None
        for i in range(X_train.shape[1]):
            if terms is None:
                terms = s(i, n_splines=self.n_splines)
            else:
                terms += s(i, n_splines=self.n_splines)
        self.model = LinearGAM(terms, max_iter=self.max_iter).fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if not _HAS_PYGAM or self.model is None:
            return float('nan')
        preds = self.model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))  # RMSE

# ------------------------------
# Test script
# ------------------------------
def test_baselines(X_train=None, X_val=None, X_test=None, y_train=None, y_val=None, y_test=None, *, seed: int = 42):
    """Train and evaluate baseline models.

    If dataset splits are provided they will be used; otherwise the default
    dataset is loaded. `seed` is keyword-only to avoid accidental positional
    collisions.
    Returns (dt_model, pygam_model, rmse_dt, rmse_gam).
    """
    # Reproducibility: set global seed for baselines
    set_global_seed(seed)

    # Load default regression dataset if splits not provided
    if X_train is None:
        data = RegressionDataset(default="california", scale=True, random_state=seed)
        X_train, X_val, X_test, y_train, y_val, y_test = data.get_splits()

    # Decision Tree
    dt = DecisionTreeRegressionBaseline(max_depth=6, random_state=seed)
    dt.fit(X_train, y_train)
    rmse_dt = dt.evaluate(X_test, y_test)
    print(f"Decision Tree RMSE: {rmse_dt:.4f}")

    # PyGAM
    gam = PyGAMRegressionBaseline(n_splines=10)
    gam.fit(X_train, y_train)
    rmse_gam = gam.evaluate(X_test, y_test)
    print(f"PyGAM RMSE: {rmse_gam:.4f}")

    return dt, gam, rmse_dt, rmse_gam


if __name__ == "__main__":
    test_baselines()

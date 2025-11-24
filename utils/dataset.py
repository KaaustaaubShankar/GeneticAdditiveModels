from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_diabetes
class RegressionDataset:
    def __init__(self, default="california", test_size=0.2, val_size=0.2, scale=True, random_state=42):
        self.default = default
        self.test_size = test_size
        self.val_size = val_size
        self.scale = scale
        self.random_state = random_state
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.scaler = None

        self._load_dataset()
        self._split()
        if self.scale:
            self._apply_scaling()

    def _load_dataset(self):
        if self.default == "california":
            data = fetch_california_housing(as_frame=True)
        else:
            raise ValueError(f"Unknown dataset '{self.default}'")
        self.X = data.frame.drop(columns=[data.target.name])
        self.y = data.target

    def _split(self):
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        val_ratio = self.val_size / (1 - self.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )

    def _apply_scaling(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def get_splits(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test



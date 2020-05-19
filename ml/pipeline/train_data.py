
from sklearn.model_selection import train_test_split

def _target(self):
    y = self._get_raw_data()["Item_Outlet_Sales"]
    return y

def train_test_data(self) -> tuple:
    # TODO: check the sorting X and y
    X = self._feature_matrix
    y = self._target
    assert X.shape[0] == y.shape[0]  # use raise instead
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=self.test_size, random_state=self.random_state
    )
    return (X_train, y_train, X_test, y_test)

def train_data(self) -> tuple:
    return self.train_test_data[:2]

@cached_property
def test_data(self) -> tuple:
    return self.train_test_data[2:]
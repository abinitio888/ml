from sklearn.model_selection import train_test_split
from ml.helper.config import Config

import pandas as pd


class Data:
    def __init__(self, config: Config):
        self.config = Config
        self.random_state = self.config.data["random_state"]
        self.test_size = self.config.data["test_size"]

    def tab_1(self):
        # spark --> spark
        pass

    def tab_2(self):
        # spark --> spark
        pass

    def _get_raw_data(self) -> pd.DataFrame:
        # spark --> pd
        # feature engineering
        raw_data = pd.DataFrame()

        X = raw_data[:, :-2]
        y = raw_data[:, -1:]
        return X, y

    @property
    def raw_data(self):
        return self._get_raw_data()

    @property
    def train_data(self) -> pd.DataFrame:
        X, y = self.raw_data
        X_train, y_train, _, _ = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, y_train

    @property
    def test_data(self) -> pd.DataFrame:
        X, y = self.raw_data
        _, _, X_test, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_test, y_test

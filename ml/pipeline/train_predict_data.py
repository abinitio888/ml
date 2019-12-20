from cached_property import cached_property

from sklearn.model_selection import train_test_split

from ml.pipeline.feature import Feature


class TrainPredictData:
    def __init__(self, config, df):
        self.config = config
        self.df = df

        self.ft_on = config["featuretools"]["on"]
        if not self.ft_on:
            self.features = self.config["features"]

        self.target = self.config["target"]
        self.test_size = self.config["train"]["test_size"]
        self.randomn_state = self.config["train"]["randomn_state"]

    @cached_property
    def _train_test_data(self) -> tuple:
        # TODO: check the sorting X and y
        if self.ft_on:
            X = Feature(self.df).feature_matrix
        else:
            X = self.df[self.features]

        y = self.df[self.target]
        assert X.shape[0] == y.shape[0]  # use raise instead

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.randomn_state
        )
        return (X_train, y_train, X_test, y_test)

    @cached_property
    def train_data(self) -> tuple:
        return self._train_test_data[:2]

    @cached_property
    def test_data(self) -> tuple:
        return self._train_test_data[2:]

    @cached_property
    def predict_data(self):
        # if self.ft_on:
        #     X = Feature(self.df).feature_matrix
        pass

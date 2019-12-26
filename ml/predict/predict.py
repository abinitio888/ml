from joblib import load
from ml.data.data import Data
from ml.helper.config import Config
import logging


class Predict:
    """
    >>> pipeline = Predict(config, data)
    >>> pipeline.predict()
    """

    def __init__(self, config: Config, data: Data):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data

        self.X, self.y = self.data.test_data
        self.model_persistent_path = self.config["io"]["model_persistent_path"]
        breakpoint()

        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        pipeline = load(self.model_persistent_path + "pipeline.joblib")
        return pipeline

    def predict(self):
        predictions = self.pipeline.predict(self.X)
        return predictions


if __name__ == "__main__":
    from ml.helper.spark_io import SparkReader

    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader)
    pipeline = Predict(config, data)
    predictions = pipeline.predict()
    print(predictions)

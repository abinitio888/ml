from shutil import rmtree
from tempfile import mkdtemp
from cached_property import cached_property
from joblib import dump, load
import logging
from typing import List
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder


from ml.data.data import Data
from ml.helper.config import Config


class Pipeline:
    """
    >>> pipeline = Pipeline(config, data) 
    >>> pipeline.run()
    """

    def __init__(self, config: Config, data: Data):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data

        self.X, self.y = self.data.train_data
        self.model_persistent_path = self.config["io"]["model_persistent_path"]
        self.top_k_features = self.config["params"]["top_k_features"]
        self.param_grid = self.config["params"]["param_grid"]

    def _make_pipeline(self):
        self.logger.info("Making the pipeline...")
        cachedir = mkdtemp()
        pipeline = make_pipeline(OneHotEncoder(), AdaBoostRegressor(), memory=cachedir)
        # pipeline = make_pipeline(RandomForestClassifier(), memory=cachedir)
        return pipeline

    def _select_features(self) -> List:
        self.logger.info("One-shot fiting and selecting the features...")
        self.pipeline.fit(self.X, self.y)
        if self.top_k_features == "auto":
            # TODO:
            # add up fi to 1
            pass
        elif type(self.top_k_features) == int:
            sorted_fi, sorted_idx = self._feature_importances
            features = self.X.columns.to_list()[-self.top_k_features :]
        return features

    # # TODO
    def _tune_hyperparams(self, features=None):
        self.logger.info("Tuning the pipeline...")
        grid_search = GridSearchCV(self.pipeline, self.param_grid)
        grid_search.fit(self.X[features], self.y)
        best_estimator = grid_search.best_estimator_
        return best_estimator

    def _dump_pipeline(self):
        self.logger.info("Dumping the pipeline...")
        # TODO: pathlib and timestamp
        dump(self.pipeline, self.model_persistent_path + "pipeline.joblib")

    def run(self):
        self.logger.info("Ready to run the pipeline:")
        self.pipeline = self._make_pipeline()
        selected_features = self._select_features()
        self._tune_hyperparams(features=selected_features)
        self._dump_pipeline()

    @cached_property
    def _feature_importances(self):
        # TO_FIX: -1 is hard-coded
        # use the robust decrease in accuracy instead
        fi = self.pipeline[-1].feature_importances_
        sorted_idx = np.argsort(fi)
        sorted_fi = fi[sorted_idx]
        return sorted_fi, sorted_idx


if __name__ == "__main__":
    from ml.helper.spark_io import SparkReader

    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader)
    pipeline = Pipeline(config, data)
    pipeline.run()
    print("test pass")

from cached_property import cached_property
from joblib import dump, load
from shutil import rmtree
from tempfile import mkdtemp
from typing import List
import logging
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import mlflow
import mlflow.sklearn

from ml.data.data import Data
from ml.helper.config import Config


class Pipeline:
    """
    >>> pipeline = Pipeline(config, df) 
    >>> pipeline.train()
    or 
    >>> pipeline.predict()
    """

    def __init__(self, config: Config, df: pd.DataFrame):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.df = df 

        self.X, self.y = self.data.train_data
        self.top_k_features = self.config["params"]["top_k_features"]
        self.param_grid = self.config["params"]["param_grid"]

        self.model_persistent_path = self.config["io"]["model_persistent_path"]
        self.mlflow_experiment_name = self.config["io"]["mlflow_experiment_name"]
        self.mlflow_tracking_uri = self.config["io"]["mlflow_tracking_uri"]

    def _make_pipeline(self):
        self.logger.info("Making the pipeline...")
        cachedir = mkdtemp()
        pipeline = make_pipeline(OneHotEncoder(), AdaBoostRegressor(), memory=cachedir)
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

    def _tune_hyperparams(self, features=None):
        self.logger.info("Tuning the pipeline...")
        grid_search = GridSearchCV(self.pipeline, self.param_grid)
        grid_search.fit(self.X[features], self.y)
        best_estimator = grid_search.best_estimator_
        # import ipdb; ipdb.set_trace()
        return best_estimator

    def _dump_pipeline(self, tuned_pipeline):
        self.logger.info("Dumping the pipeline...")
        # TODO: pathlib and timestamp
        dump(tuned_pipeline, self.model_persistent_path + "pipeline.joblib")


    def _log_pipeline(self, tuned_pipeline):
        """
        mlflow run: no use
        mlflow models serve: no use, focus on offline batch predition

        TODO: Launching Multiple Runs in One Program, mlflow.start_run()
        """
        self.logger.info("Logging the pipeline...")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        mlflow.log_params(tuned_pipeline[-1].get_params())
        # mlflow.log_artifact("abc")
        # mlflow.log_metrics("abc")
        # TODO: vs the dump
        mlflow.sklearn.log_model(sk_model=tuned_pipeline, artifact_path="my_pipeline", registered_model_name="boost_model")

    @cached_property
    def _feature_importances(self):
        # TO_FIX: -1 is hard-coded
        # use the robust decrease in accuracy instead
        fi = self.pipeline[-1].feature_importances_
        sorted_idx = np.argsort(fi)
        sorted_fi = fi[sorted_idx]
        return sorted_fi, sorted_idx

    def train(self):
        self.logger.info("Ready to run the pipeline:")
        self.pipeline = self._make_pipeline()
        selected_features = self._select_features()
        tuned_pipeline = self._tune_hyperparams(features=selected_features)
        self._dump_pipeline(tuned_pipeline)
        self._log_pipeline(tuned_pipeline)
        self.logger.info("The pipeline is finished")

    def predict(self):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data = data

        self.X, self.y = self.data.test_data
        self.model_persistent_path = self.config["io"]["model_persistent_path"]

        self.pipeline = self._load_pipeline()
        # mlflow native model loader
        # mlflow.sklearn.load_model()
        pipeline = load(self.model_persistent_path + "pipeline.joblib")
        predictions = self.pipeline.predict(self.X)
        return predictions


if __name__ == "__main__":
    # from ml.helper.spark_io import SparkReader

    # config = Config("./ml/confs/").config
    # spark_reader = SparkReader(config)
    # data = Data(config, spark_reader)
    # pipeline = Pipeline(config, data)
    # pipeline.train()
    print("test pass")

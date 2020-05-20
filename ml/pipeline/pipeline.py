import logging
import numpy as np
import pandas as pd

from cached_property import cached_property
from joblib import dump, load
from shutil import rmtree
from tempfile import mkdtemp
from typing import List

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import mlflow
import mlflow.sklearn

from ml.data.data import Data
from ml.helper.config import Config
from ml.pipeline.feature_selection import FeatureSelection
from ml.pipeline.hyperparam import HyperParam
from ml.pipeline.train_predict_data import TrainPredictData


class Pipeline:
    """
    >>> pipeline = Pipeline(config, df, is_train=True) 
    >>> pipeline.train()
    or 
    >>> pipeline = Pipeline(config, df, is_train=False) 
    >>> pipeline.predict()
    """

    def __init__(self, config: Config, df: pd.DataFrame, is_train=True):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.is_train = is_train
        self.df = df
        # TODO: needs to be improved
        self.model_persistent_path = self.config["io"]["model_persistent_path"]
        self.mlflow_experiment_name = self.config["mlflow"]["mlflow_experiment_name"]
        self.mlflow_tracking_uri = self.config["mlflow"]["mlflow_tracking_uri"]

    def _make_pipeline(self):
        cachedir = mkdtemp()
        # TODO: the pipeline is hard-coded
        self.pipeline = make_pipeline(
            OneHotEncoder(), AdaBoostRegressor(), memory=cachedir
        )
        self.logger.info("Pipeline is made.")

    def _prepare_data(self):
        self.data = TrainPredictData(self.config, self.df)
        if self.is_train:
            self.X, self.y = self.data.train_data
        else:
            self.X = self.data.predict_data

    def _select_features(self):
        self.features = FeatureSelection(self.config, self.data, self.pipeline)
        self.logger.info("Features are selected.")

    def _tune_hyperparams(self):
        self.best_pipeline = HyperParam(self.config, self.data, self.pipeline).tune()
        self.logger.info("Hyper-parameter tuning is finished.")

    def _log_pipeline(self):
        """
        mlflow run: no use
        mlflow models serve: no use, focus on offline batch predition

        TODO: Launching Multiple Runs in One Program, mlflow.start_run()
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        mlflow.log_params(self.best_pipeline[-1].get_params())
        # mlflow.log_artifact("abc")
        # mlflow.log_metrics("abc")
        mlflow.sklearn.log_model(
            sk_model=self.best_pipeline,
            artifact_path="my_pipeline",
            registered_model_name="boost_model",
        )
        self.logger.info("The pipeline is logged.")

    def train(self):
        self.logger.info("Ready to train the pipeline:")
        self._prepare_data()
        self._make_pipeline()
        self._select_features()
        self._tune_hyperparams()
        self._log_pipeline()
        self.logger.info("The train is finished")

    def predict(self):
        self.logger.info("Ready to predit:")
        self._prepare_data()
        pipeline = mlflow.sklearn.load_model(
            self.model_persistent_path + "pipeline.joblib"
        )
        pred = pipeline.predict(self.X)
        self.logger.info("The predict is finished")
        return pred


if __name__ == "__main__":
    from ml.helper.spark_io import SparkReader

    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader)
    pipeline = Pipeline(config, data, is_train=True)
    pipeline.train()
    pipeline.predict()
    print("test pass")

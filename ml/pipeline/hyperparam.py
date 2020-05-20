import logging
import numpy as np

from cached_property import cached_property
from typing import List

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ml.helper.config import Config
from ml.pipeline.train_predict_data import TrainPredictData


class HyperParam:
    def __init__(self, config: Config, data: TrainPredictData, pipeline: Pipeline):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pipeline = pipeline
        self.data = data
        self.param_grid = self.config["train"]["params"]["model_params"]["param_grid"]

    def tune(self):
        self.X, self.y = self.data.train_data
        grid_search = GridSearchCV(self.pipeline, self.param_grid)
        features = self.pipeline.features
        grid_search.fit(self.X[features], self.y)
        best_pipeline = grid_search.best_estimator_
        return best_pipeline

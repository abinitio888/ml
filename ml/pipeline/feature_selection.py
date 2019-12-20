import logging
import numpy as np

from cached_property import cached_property
from sklearn.pipeline import Pipeline
from typing import List

from ml.helper.config import Config
from ml.pipeline.train_predict_data import TrainPredictData


class FeatureSelection:
    def __init__(self, config: Config, data: TrainPredictData, pipeline: Pipeline):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pipeline = pipeline
        self.data = data

        self.top_k_features = self.config["train"]["params"]["model_params"][
            "top_k_features"
        ]

    @cached_property
    def features(self) -> List:
        self.X, self.y = self.data.train_data
        self.pipeline.fit(self.X, self.y)
        self.logger.info("One-shot fitting is finished.")
        if self.top_k_features == "auto":
            # TODO:
            # add up fi to 1
            pass
        elif type(self.top_k_features) == int:
            sorted_fi, sorted_idx = self._feature_importances
            # TODO: Bug here
            features = self.X.columns.to_list()[-self.top_k_features :]
        self.logger.info("Features selection is finished.")
        return features

    @cached_property
    def _feature_importances(self):
        # TO_FIX: -1 is hard-coded
        # use the robust decrease in accuracy instead
        fi = self.pipeline[-1].feature_importances_
        sorted_idx = np.argsort(fi)
        sorted_fi = fi[sorted_idx]
        self.logger.info("Feature importance is calculated.")
        return sorted_fi, sorted_idx

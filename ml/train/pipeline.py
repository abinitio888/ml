from shutil import rmtree
from tempfile import mkdtemp
from cached_property import cached_property
from joblib import dump, load
import logging

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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

    def _make_pipeline(self):
        self.logger.info("Making the pipeline...")
        cachedir = mkdtemp()
        pipeline = make_pipeline(RandomForestClassifier(), memory=cachedir)
        return pipeline

    # # TODO
    def _tune(self):
        self.logger.info("Tuning the pipeline...")
        pass

    #     param_grid = dict()
    #     grid_search = GridSearchCV(self.pipeline, param_grid)
    #     return grid_search
    # grid_search = self._tune_hyperparam()
    # results = grid_search.fit(self.X, self.y)
    # return results

    def _fit(self):
        self.logger.info("Fiting the pipeline...")
        self.pipeline.fit(self.X, self.y)

    def _dump_pipeline(self):
        self.logger.info("Dumping the pipeline...")
        # TODO: pathlib and timestamp
        dump(self.pipeline, self.model_persistent_path + "pipeline.joblib")

    def run(self):
        self.logger.info("Ready to run the pipeline:")
        self.pipeline = self._make_pipeline()
        self._tune()
        self._fit()
        self._dump_pipeline()

    @cached_property
    def feature_importances(self):
        # TO_FIX: -1 is hard-coded
        fi = self.pipeline[-1].feature_importances_
        return fi


if __name__ == "__main__":
    from ml.helper.spark_io import SparkReader

    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader)
    pipeline = Pipeline(config, data)
    pipeline.run()

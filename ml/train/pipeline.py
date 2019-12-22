from shutil import rmtree
from tempfile import mkdtemp

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from ml.data.data import Data
from ml.helper.config import Config


class PipeLine:
    def __init__(self, config: Config, data: Data):
        self.config = config
        self.data = data
        self.X, self.y = self.data.train_data

    def _make_pipeline(self):
        cachedir = mkdtemp()
        pipeline = make_pipeline(PCA(), SVC(), memory=cachedir)
        return pipeline

    def _tune_hyperparam(self):
        pipeline = self._make_pipeline()

        param_grid = dict()
        grid_search = GridSearchCV(pipeline, param_grid)
        return grid_search

    def fit(self):
        grid_search = self._tune_hyperparam()

        results = grid_search.fit(self.X, self.y)
        return results

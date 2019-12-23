import pandas as pd
import pyspark.sql as ps

import logging
from typing import Set
from cached_property import cached_property

from sklearn.model_selection import train_test_split

from ml.data.data_streams import DataStreams
from ml.helper.config import Config
from ml.helper.spark_io import SparkReader


class Data:
    def __init__(self, config: Config, spark_reader: SparkReader):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.spark_reader = spark_reader
        self.random_state = self.config["data"]["random_state"]
        self.test_size = self.config["data"]["test_size"]

        self._append_streams()

    def _append_streams(self):
        self._streams = DataStreams(self.config, self.spark_reader)
        for stream_name, stream in self._streams.streams.items():
            setattr(self, stream_name, stream)

    @cached_property
    def _df_articles(self):
        # spark --> spark
        df = self.article_datamart.df
        return df

    @cached_property
    def _df_clf(self):
        # spark --> spark
        df = self.clf_datamart.df
        return df

    def tab_2(self):
        # spark --> spark
        pass

    def _get_raw_data(self) -> pd.DataFrame:
        # spark --> pd
        # feature engineering
        raw_data = self._df_clf

        raw_data = raw_data.toPandas()

        X = raw_data.loc[:, ["x1", "x2", "x3", "x4"]]
        y = raw_data.loc[:, "y"]
        return X, y

    @cached_property
    def train_test_data(self) -> tuple:
        X, y = self._get_raw_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return (X_train, y_train, X_test, y_test)

    @cached_property
    def train_data(self) -> tuple:
        return self.train_test_data[:2]

    @cached_property
    def test_data(self) -> tuple:
        return self.train_test_data[2:]


if __name__ == "__main__":
    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader)
    print(data.train_data[0].shape)
    print(data.train_data[1].shape)
    print(data.test_data[0].shape)
    print(data.test_data[1].shape)

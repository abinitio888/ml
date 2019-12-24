import pandas as pd
import pyspark.sql as ps

import logging
from typing import Set
from cached_property import cached_property
import featuretools as ft

from sklearn.model_selection import train_test_split

from ml.data.data_streams import DataStreams
from ml.helper.config import Config
from ml.helper.spark_io import SparkReader

# refactor the featuretools code to make it more general


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

    @cached_property
    def _df_bigmart(self):
        # spark --> spark
        df = self.bigmart_datamart.df
        return df

    def tab_tmp(self):
        # spark --> spark
        pass

    def _get_raw_data(self) -> pd.DataFrame:
        # spark --> pd
        # table manipulation to suit the feature engineer
        raw_data = self._df_bigmart

        raw_data = raw_data.toPandas()
        raw_data["id"] = raw_data["Item_Identifier"] + raw_data["Outlet_Identifier"]
        raw_data.drop(["Item_Identifier"], axis=1, inplace=True)
        return raw_data

    def _construct_entity_set(self):
        raw_data = self._get_raw_data()

        es = ft.EntitySet(id="sales")
        es.entity_from_dataframe(entity_id="bigmart", dataframe=raw_data, index="id")

        es.normalize_entity(
            base_entity_id="bigmart",
            new_entity_id="outlet",
            index="Outlet_Identifier",
            additional_variable=[
                "Outlet_Establishment_Year",
                "Outlet_Size",
                "Outlet_Location_Type",
                "Outlet_Type",
            ],
        )

    def _get_feature_matrix(self):
        es = self._construct_entity_set()
        feature_matrix, feature_defs = ft.dfs(
            entityset=es, target_entity="bigmart", max_depth=2, verbose=1, n_jobs=1
        )

        feature_matrix.drop(["Outlet_Identifier"], axis=1, inplace=True)

        return feature_matrix

    def _get_target(self):
        y = self._get_raw_data["Item_Outlet_Sales"]
        return y

    @cached_property
    def train_test_data(self) -> tuple:
        # TODO: check the sorting X and y
        X = self._get_feature_matrix()
        y = self._get_target()
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

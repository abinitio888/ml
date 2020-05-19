import featuretools as ft
from cached_property import cached_property
import logging


class Feature:
    """
    This module acts on the pandas DataFrame.
    This module will hard-code the feature generation for each use case.
    This module should be used for both train and predict. That's why I put in the pipeline folder
    """
    def __init__(self, df):
        self.logger = logging.getLogger(__name__)
        self.df = df
        fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}
        self.df['Item_Fat_Content'] = self.df['Item_Fat_Content'].replace(fat_content_dict, regex=True)
        self.df["id"] = self.df["Item_Identifier"] + self.df["Outlet_Identifier"]
        self.df.drop(["Item_Identifier"], axis=1, inplace=True)
        self.df = self.df.dropna(axis=0)

    def _construct_entity_set(self):
        """
        Construct and dump the EntitySet
        """
        es = ft.EntitySet(id="sales")
        es.entity_from_dataframe(entity_id="bigmart", dataframe=self.df, index="id")

        es.normalize_entity(
            base_entity_id="bigmart",
            new_entity_id="outlet",
            index="Outlet_Identifier",
            additional_variables=[
                "Outlet_Establishment_Year",
                "Outlet_Size",
                "Outlet_Location_Type",
                "Outlet_Type",
            ],
        )

        # TODO: dump the es file for future featuretools
        print(es)
        return es

    def _format_feature_matrix(self, feature_matrix):
        # categorical_features = np.where(feature_matrix.dtypes == 'object')[0]
        # for i in categorical_features:
        #     feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')
        feature_matrix.drop(["Outlet_Identifier"], axis=1, inplace=True)
        return feature_matrix

    @cached_property
    def feature_matrix(self):
        """
        Use `featuretools` to generate the `feature_matrix` and
        format the feature matrix according to the model
        """
        es = self._construct_entity_set()
        feature_matrix, feature_defs = ft.dfs(
            entityset=es, target_entity="bigmart", max_depth=2, verbose=1, n_jobs=1
        )

        new_feature_matrix = self._format_feature_matrix(feature_matrix)
        self.logging.info("Auto feature generation is finished.")
        return new_feature_matrix


if __name__ == "__main__":
    from ml.helper.config import Config
    from ml.helper.spark_io import SparkReader
    from ml.data.data import Data

    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader, train=True)
    df = data.master_df.toPandas()[: 1000]
    # import ipdb; ipdb.set_trace()
    df_new = Feature(df).feature_matrix
    print(df_new.head())
    print("test passed!")
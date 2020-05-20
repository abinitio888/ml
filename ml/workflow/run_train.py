import pandas as pd
from pyspark.sql.functions import current_date

from ml.data.data import Data
from ml.pipeline.feature import Feature
from ml.helper.config import Config
from ml.pipeline.pipeline import Pipeline
from ml.pipeline.train_predict_data import TrainPredictData


@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def train(df: pd.DataFrame) -> pd.DataFrame:
    config = Config("./ml/confs/").config

    ft_on = config["featuretools"]["on"]
    if ft_on: 
        feature_matrix = Feature(df).feature_matrix
        train_predict_data = TrainPredictData(config, df, feature_matrix=feature_matrix)
    else:
        train_predict_data = TrainPredictData(config, df, feature_matrix=None)

    pipeline = Pipeline(config, train_predict_data)
    pipeline.train()

    # This is needed for spark to collect the results cross the workers.
    result = Result()
    return result

config = Config("./ml/confs/").config
data = Data(config, spark_reader, train=True)
results = (
        data.master_df
        .groupBy("scenario_id")
        .apply(train)
        .withColumn("model_train_date", current_date())
        )


import pandas as pd
from pyspark.sql.functions import current_date

from ml.data.data import Data
from ml.train.features import Features
from ml.helper.config import Config
from ml.train.pipeline import Pipeline
from ml.predict.predict import Predict

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def predict(df: pd.DataFrame) -> pd.DataFrame:
    config = Config("./ml/confs/").config

    feature_matrix = Feature(df)

    pipeline = Predict(config, feature_matrix)
    pipeline.predict()

    # This is needed for spark to collect the results cross the workers.
    result = Result()
    return result

config = Config("./ml/confs/predict/").config
data = Data(config, spark_reader)
results = (
        data.get_master_df()
        .groupBy("scenario_id")
        .apply(predict)
        .withColumn("model_predict_date", current_date())
        )
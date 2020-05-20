import pandas as pd
from pyspark.sql.functions import current_date

from ml.data.data import Data
from ml.helper.config import Config
from ml.pipeline.pipeline import Pipeline


@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def train(df: pd.DataFrame) -> pd.DataFrame:
    config = Config("./ml/confs/").config
    pipeline = Pipeline(config, df, is_train=True)
    pipeline.train()

    # This is needed for spark to collect the results cross the workers.
    result = Result()
    return result


config = Config("./ml/confs/").config
data = Data(config, spark_reader, is_train=True)
results = (
    data.master_df.groupBy("scenario_id")
    .apply(train)
    .withColumn("model_train_date", current_date())
)

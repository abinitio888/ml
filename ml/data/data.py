import logging
import time
from cached_property import cached_property
from pyspark.sql import DataFrame 

from ml.data.data_streams import DataStreams
from ml.helper.config import Config
from ml.helper.spark_io import SparkReader

# TODO: how to handle predict data, which doesn't have y
# for predict, do not read the y datastream?


class Data:
    def __init__(self, config: Config, spark_reader: SparkReader, train=True):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.spark_reader = spark_reader
        self.datalake_url = self.config["io"]["datalake_url"]
        self.train_data_path = self.config["io"]["train_data_path"]
        self.predict_data_path = self.config["io"]["predict_data_path"]
            
        self._append_streams()
        self.train = train

    def _append_streams(self):
        self._streams = DataStreams(self.config, self.spark_reader)
        for stream_name, stream in self._streams.streams.items():
            setattr(self, stream_name, stream)

    @cached_property
    def _df_bigmart(self) -> DataFrame:
        df = self.bigmart_datamart.df
        return df

    @cached_property
    def master_df(self) -> DataFrame:
        """
        Master dataframe, where dataframe operations happen.
        """
        master_df = self._df_bigmart

        # TODO: switch to delta
        if self.train:
            master_df.write.format("parquet").save(self.datalake_url + self.train_data_path + time.strftime("%Y%m%d%H%M%S"))
            self.logger.info("Train master data is fetched and saved.")
        else:
            master_df.write.format("parquet").save(self.datalake_url + self.predict_data_path + time.strftime("%Y%m%d%H%M%S"))
            self.logger.info("Predict master data is fetched and saved.")
        return master_df 


if __name__ == "__main__":
    config = Config("./ml/confs/").config
    spark_reader = SparkReader(config)
    data = Data(config, spark_reader, train=True)
    df = data.master_df
    df.printSchema()
    df.groupBy("scenario_id").count().show()
    print("test passed!")

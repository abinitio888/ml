import abc
import logging
from pyspark.sql import DataFrame


class DataStream:
    def __init__(self, stream_name, config, spark_reader):
        self.logger = logging.getLogger(__name__)
        self.stream_name = stream_name
        self.config = config
        self.spark_reader = spark_reader

        self._set_stream_attributes()

    def _set_stream_attributes(self):
        for key, value in self.config["data_sources"][self.stream_name].items():
            setattr(self, key, value)

    @property
    def df(self) -> DataFrame:
        df = self.spark_reader.read(
            stream_name=self.stream_name,
            columns=self.columns,
            source=self.source,
            extension=self.extension,
            location=self.location,
        )        
        self.logger.info(f"Data stream: {self.stream_name} is fetched.")
        return df
        
        
class DataStreams:
    def __init__(self, config, spark_reader):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.spark_reader = spark_reader

        self.streams = {}
        self._set_streams()

    def _set_streams(self):
        for stream_name, _ in self.config["data_sources"].items():
            stream = DataStream(stream_name, self.config, self.spark_reader)
            self.streams[stream_name] = stream

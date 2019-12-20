import abc
import os
import yaml


class Config:
    def __init__(self, config_folder_path=None):
        self.path = config_folder_path

    @property
    def config(self):
        config = {}
        yml_files = self._get_yaml_files()

        for f in yml_files:
            with open(f, "r") as fd:
                yaml_dict = yaml.load(fd, Loader=yaml.FullLoader) or {}
                config.update(yaml_dict)
        return config

    def _get_yaml_files(self):
        yml_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(self.path)
            for name in files
            if name.endswith(".yml")
        ]
        return yml_files


if __name__ == "__main__":
    config = Config("./ml/confs/").config
    print(config.keys())

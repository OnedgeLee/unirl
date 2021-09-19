import os
import yaml

class YamlConfig(dict):

    def __init__(self, config_paths: dict):

        for key, path in config_paths.items():
            self.__setitem__(key, self._yaml_to_config_dict(path))

    @staticmethod
    def _yaml_to_config_dict(path: str) -> dict:

        try:
            with open(path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            with open(os.path.expanduser(path)) as f:
                data = yaml.load(f, loader=yaml.FullLoader)

        return data

    


    
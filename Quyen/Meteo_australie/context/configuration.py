# -*- coding: utf-8 -*-
"""

"""

import yaml
import os


class Configuration:
    """Configuration de la commande"""

    def __init__(self) -> None:
        super().__init__()
        self._conf = None

    def initialize(self, file_path: str) -> None:
        if self._conf is None:
            conf_file_path = os.path.abspath(file_path)
            with open(conf_file_path, "r") as ymlfile:
                self._conf = yaml.load(ymlfile, Loader=yaml.FullLoader)

    def get_full(self):
        if self._conf is None:
            raise EnvironmentError("Configuration : Configuration non charge")
        return self._conf

    def get_value(self, *keys):
        if self._conf is None:
            raise EnvironmentError("Configuration : Configuration non charge")
        val = self._conf
        for key in list(keys):
            val = val[key]
        return val

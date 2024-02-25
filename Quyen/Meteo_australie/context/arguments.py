# -*- coding: utf-8 -*-
"""

"""

import sys
import getopt


class Arguments:
    """Arguments de la commande"""

    def __init__(self) -> None:
        ###
        # options
        ###
        self._opt_short = "avwc:"
        self._opt_long = ["not-remove-log", "verbose-moy", "verbose-max", "config-path="]
        ###
        # valeurs arguments
        ###
        self._opt_values: list = []
        self._param_values: list = []
        self._param_count: int = 0
        ###
        # variable
        ###
        self._opt_permanent = False
        self._opt_verbose = 0
        self._opt_config_path = ".config.yml"

    def initialize(self) -> None:
        # gestion des options et parametres
        args_full = sys.argv
        args_opt_param = args_full[1:]
        self._opt_values, self._param_values = getopt.getopt(args_opt_param, self._opt_short, self._opt_long)
        self._param_count = len(self._param_values)
        for opt, val in self._opt_values:
            if opt in ('-a', '--not-remove-log'):
                self._opt_permanent = True
            if opt in ('-v', '--verbose-moy'):
                self._opt_verbose = 50
            if opt in ('-w', '--verbose-max'):
                self._opt_verbose = 100
            if opt in ('-c', '--config-path'):
                self._opt_config_path = val

    def get_opt_values(self) -> []:
        return self._opt_values

    def get_param_values(self) -> []:
        return self._param_values

    def is_permanent(self) -> bool:
        return self._opt_permanent

    def get_verbose(self) -> int:
        return self._opt_verbose

    def get_config_path(self) -> str:
        return self._opt_config_path

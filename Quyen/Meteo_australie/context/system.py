
# -*- coding: utf-8 -*-
"""Module pour recupérer des informations systeme sur le process en cours"""

import os
import sys

import psutil


if not (psutil.LINUX or psutil.MACOS or psutil.WINDOWS):
    sys.exit("platform not supported")

__all__ = ['get_memory_info', 'get_memory_full_info', 'get_memory_uss', 'get_memory_rss']


def convert_bytes_to_string(n: int) -> str:
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {s: 1 << (i + 1) * 10 for i, s in enumerate(symbols)}
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n


process = psutil.Process(os.getpid())


def get_memory_info() -> str:
    """
        Retourne somme la mémoire USS, PSS (partagé) et SWAP
        :return: str
    """
    meminfo = process.memory_info()
    return str(meminfo)


def get_memory_full_info() -> str:
    meminfo = process.memory_full_info()
    return str(meminfo)


def get_memory_uss() -> str:
    """
        Retourne la memoire USS en Mo (L'USS (Unique Set Size) est la mémoire qui est unique à un processus
        et qui serait libérée si le processus était terminé maintenant)
        :return: str
    """
    meminfo = process.memory_full_info()
    return convert_bytes_to_string(meminfo.uss)


def get_memory_rss() -> str:
    """
        Retourne la memoire RSS en Mo (Memoire process + virtuelle + partagé)
        :return: str
    """
    meminfo = process.memory_full_info()
    return convert_bytes_to_string(meminfo.rss)


if __name__ == '__main__':
    print(os.name)
    print(get_memory_uss())

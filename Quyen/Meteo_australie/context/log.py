# -*- coding: utf-8 -*-
"""Module de context pour la generation de log de calcul"""

__all__ = ['CtxLog', 'Verbose']

import os
import sys
import time
import traceback
import uuid

import psutil

from context import timer, system

if not (psutil.LINUX or psutil.MACOS or psutil.WINDOWS):
    raise EnvironmentError("Platform not supported")

__all__ = ['CtxLog', 'Verbose']

LOG_LINE = '----------------------------------------'

_DIR_LOGS: str = "logs"
_DATETIME_FORMAT_FILE_NAME: str = "%Y%m%d-%H%M%S-"
_DATETIME_FORMAT_LOG_LINE: str = "%Y/%m/%d %H:%M:%S ## "
_TIME_FORMAT_LOG_LINE: str = "%H:%M:%S ## "

_VERBOSE_LEVEL_MIN = 0
_VERBOSE_LEVEL_MOY = 50
_VERBOSE_LEVEL_MAX = 100

UNKNOWN_ERROR = -1


class Verbose:
    level: int = 0

    def __init__(self):
        self.level = _VERBOSE_LEVEL_MIN

    def set_verbose(self, level: int):
        self.level = level

    def is_min(self) -> bool:
        return self.level >= _VERBOSE_LEVEL_MIN

    def is_moy(self) -> bool:
        return self.level >= _VERBOSE_LEVEL_MOY

    def is_max(self) -> bool:
        return self.level >= _VERBOSE_LEVEL_MAX


class CtxLog:
    """
        Classe de gestion de log de calcul
        Creation d'un fichier de log unique à chaque calcul
    """

    def __init__(self, cmd_name: str, path_log_name: str = ".", file_log_name: str = None) -> None:
        """
            Initialisation d'une instance de la classe CtxLog
            :param cmd_name: str
            :param path_log_name: str
        """
        # creation d'une instance CtxTime
        self.__tm = timer.CtxTimer()
        # _verbose : Valeur détanminant si un log doit être écrit dans le fichier de log
        #           Valeur comprise entre -1 et 100
        #           Valeur par defaut 0
        # Creation d'un instance de Verbose
        self._verbose = Verbose()
        # Nom de la commande en cours de traitement
        self._cmd_name = cmd_name
        # Construction du chemin d'acces du dossier de log
        # _path_log_name : Nom complet du dossier de log
        if path_log_name == ".":
            self._path_log_name = os.path.abspath(os.sep.join([path_log_name, _DIR_LOGS]))
        else:
            self._path_log_name = os.path.abspath(path_log_name)
        # Construction du chemain d'acces au fichier de log du traitement.
        # Nom complet du fichier de log
        if file_log_name:
            self._full_log_name = os.sep.join([self._path_log_name, file_log_name])
        else:
            self._full_log_name = os.sep.join([self._path_log_name, self._create_name_file()])
        # Creation des dossiers si necessaires et ouverture
        os.makedirs(self._path_log_name, exist_ok=True)

        self.__fsock = open(self._full_log_name, 'x', encoding="utf-8")

    def __del__(self) -> None:
        """
            Destructeur d'un instance de la classe CtxLog
        """
        if self.__fsock is not None:
            self.__fsock.close()
        # print("--- fichier de log fermé", self._full_log_name, "---")

    @staticmethod
    def _create_name_file() -> str:
        """
            retourn la creation d'un nom de fichier unique pour les log de traitement
            :return: str
        """
        tm = time.localtime(time.time())
        return time.strftime(_DATETIME_FORMAT_FILE_NAME, tm) + str(uuid.uuid4()).replace("-", "") + ".log"

    @staticmethod
    def _get_datetime_log_line() -> str:
        """
            retourne la date et l'heure pour une ligne de log
            :return: str
        """
        t = time.time()
        tm = time.localtime(t)
        return time.strftime(_DATETIME_FORMAT_LOG_LINE, tm)

    @staticmethod
    def _get_time_log_line() -> str:
        """
            retourne l'heure pour une ligne de log
            :return: str
        """
        t = time.time()
        tm = time.localtime(t)
        return time.strftime(_TIME_FORMAT_LOG_LINE, tm)

    def _log(self, *xargs, xsep=' ', xend='\n') -> None:
        """
            Ecriture dans le fichier _log avec Time
            :param args: liste de données à ecrire
            :param sep: str
            :param end: str
            :return: None
        """
        # preparation si le premier argument est un tuple
        if type(xargs[0]).__name__ == 'tuple':
            targs, tsep, tend = xargs[0]
        else:
            targs = xargs
            tsep = xsep
            tend = xend
        # ecriture
        self.__fsock.write(self._get_time_log_line())
        for idx, val in enumerate(targs):
            if idx > 0:
                self.__fsock.write(tsep + str(val))
            else:
                self.__fsock.write(str(val))
        if tend != '':
            self.__fsock.write(tend)

    def verbose(self, level: int):
        self._verbose.set_verbose(level)

    def show(self) -> None:
        """
            Ecriture dans le fichier de log des variables de l'instance
            :return: None
        """
        if self._verbose.is_min():
            self._log("----- CtxLog()")
            self._log("_cmd_name      :", self._cmd_name)
            self._log("_path_log_name :", self._path_log_name)
            self._log("_full_log_name :", self._full_log_name)
            self._log("_verbose       :", str(self._verbose.level))

    def timer_start(self, with_cout: bool = False) -> None:
        """
            Ecriture dans le fichier de log, du message de debut du traitement
            :param with_cout: bool (defaut : False)
            :return: None
        """
        if self._verbose.is_min():
            m = "debut du traitement de '" + self._cmd_name + "', le " + self.__tm.start_to_string()
            # ecriture
            if with_cout:
                self.memory_uss(with_cout)
                self._log(sys.argv)
            else:
                self._log(m)
                self.memory_uss(with_cout)
                self._log(sys.argv)
                self._log(LOG_LINE)

    def timer_step(self, message: str = '', with_cout: bool = False) -> None:
        """
            Ecriture dans le fichier de log, d'une etape du traitement
            :param message : str (default : '')
            :param with_cout: bool (defaut : False)
            :return: None
        """
        if self._verbose.is_min():
            if len(message) > 0:
                m = message
            else:
                m = "étape en "
            m += self.__tm.step_to_string()
            # ecriture
            if with_cout:
                self.memory_uss(with_cout)
            else:
                self.memory_uss(with_cout)
                self._log(m)

    def timer_stop(self, message: str = '', error_num: int = 0, with_cout: bool = False) -> None:
        """
            Ecriture dans le fichier de log, du message de fin du traitement
            :param message : str (default : '')
            :param error_num: int (default : 0)
            :param with_cout: bool (defaut : False)
            :return: None
        """
        if self._verbose.is_min():
            if len(message) > 0:
                m = message
            else:
                m = "fin du traitement en "
            m += self.__tm.stop_to_string()
            if error_num != 0:
                m += ", error : " + str(error_num)
            # ecriture
            if with_cout:
                self.memory_uss(with_cout)
            else:
                self._log(LOG_LINE)
                self.memory_uss(with_cout)
                self._log(m)

    def log_min(self, *args, sep=' ', end='\n'):
        if self._verbose.is_min():
            self._log(args, sep, end)

    def log_moy(self, *args, sep=' ', end='\n'):
        if self._verbose.is_moy():
            self._log(args, sep, end)

    def log_max(self, *args, sep=' ', end='\n'):
        if self._verbose.is_max():
            self._log(args, sep, end)

    def __log_exception(self):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if self._verbose.is_min():
            # affichage du traceback
            tb = traceback.format_exception(exc_type, exc_obj, exc_tb)
            trace = LOG_LINE + "exception\n"
            for line in tb:
                trace += line
            self._log(trace[:-1])
        return exc_obj

    def log_err(self):
        # log exception (traceback)
        self.__log_exception()
        # prepare le code erreur et le message d'erreur
        exc_type, exc_obj, exc_tb = sys.exc_info()
        err_num = UNKNOWN_ERROR
        err_msg = exc_type.__name__ + " :: " + exc_obj.__str__()
        return err_num, err_msg

    def log_remove(self) -> None:
        if self.__fsock is not None:
            # ferme le fichier
            self.__fsock.close()
            self.__fsock = None
            # suppression du fichier de log
            try:
                os.remove(self._full_log_name)
            except:
                pass

    def log_memory_info(self, method, label, with_cout=False):
        """Utility function to log memory information."""
        message = f"{label} : {getattr(system, method)()}"
        if with_cout:
            print(message)
        else:
            self._log(message)

    # Update the methods to use the new utility functions
    memory_uss = lambda self, with_cout=False: self.log_memory_info('get_memory_uss', 'Memoire USS', with_cout)
    memory_rss = lambda self, with_cout=False: self.log_memory_info('get_memory_rss', 'Memoire RSS', with_cout)
    memory_full = lambda self, with_cout=False: self.log_memory_info('get_memory_full_info', 'Memoire ALL', with_cout)


if __name__ == '__main__':
    sys.tracebacklimit = None
    try:
        l = CtxLog("testCmd",
                   path_log_name="./fichier_log/")
        l.timer_start()
        l.show()
        time.sleep(1)
        l.log_level('moy', "log moy")
        l.log_level('max', "log max")
        l.memory_uss()
        l.memory_rss()
        l.memory_full()
        l.timer_stop()
    # except Exception as err:
    #     exception_type = type(err).__name__
    #     print(exception_type, " : ", err, sep='')
    except:
        e = sys.exc_info()[0]
        exception_type = type(e).__name__
        print(exception_type, " : ", e, sep='')

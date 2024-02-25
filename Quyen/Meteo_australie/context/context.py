# -*- coding: utf-8 -*-
"""

"""

import os
import sys
import context.log as llog


class Context:
    """Context de base de la commande"""

    def __init__(self) -> None:
        # nom du fichier avec son extention et son path
        self._cmd_path = sys.argv[0]
        # dossier contenant la commande (dernier dossier du path)
        self._cmd_folder = os.path.split(os.path.split(self._cmd_path)[0])[1]
        # nom du fichier avec son extension
        self._cmd_file = os.path.basename(self._cmd_path)
        # nom de l acommande sans son extension
        self._cmd_name = self._cmd_folder + "." + os.path.splitext(self._cmd_file)[0]
        # erreur du traitement
        self._err_num = 0
        self._err_msg = "ok"
        # Fichier de log du traitement
        self._log = None
        # reponse du traitement
        self.response = "?????"

    def create_response(self) -> str:
        """Creation de la reponse à retourner"""
        if self._err_num != 0:
            return '{"error": ' + str(self._err_num) + ', "message": "' + self._err_msg + '"}'
        else:
            return '{"error": ' + str(self._err_num) + ', "message": "' + self._err_msg + '", "data": "' + self.response + '"}'

    ###
    # Gestion des erreurs
    ###
    def get_err_num(self):
        return self._err_num

    def get_err_msg(self):
        return self._err_msg

    ###
    # Gestion des logs de traitement
    ###
    def cmd_start(self, path_log_name: str = ".", file_log_name: str = None,
                  with_cout: bool = False, verbose_level="min") -> None:
        # creation log si necessaire
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name, path_log_name=path_log_name, file_log_name=file_log_name)
            self._log.show()
        self._log.timer_start(with_cout)
        # # initialisation des variables de context
        if verbose_level == "min":
            self._log.verbose(0)
        elif verbose_level == "moy":
            self._log.verbose(50)
        elif verbose_level == "max":
            self._log.verbose(100)
        # trace des variables de context
        self.cmd_start_trace()
        self.log_min(llog.LOG_LINE)

    # def cmd_start_initialize(self) -> None:  # overwrite this function
    #     raise NotImplementedError("lib_api/context::Context::cmd_start_initialize(self)")

    def cmd_start_trace(self) -> None:
        # traces le path name de la commande
        self.log_min(sys.argv[0])

    def cmd_step(self, message: str = '', with_cout: bool = False) -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        self._log.timer_step(message, with_cout)

    def cmd_stop(self) -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        self._log.timer_stop()
        # creation dela reponse
        resp = self.create_response()
        # reponse à la commande
        self.log_min(resp)
        # supression du fichier de log si pas d'erreur et l'option permanent à False
        if self._err_num == 0 and not self.log_is_permanet():
            self.log_remove()
        # affichage de la reponse et exit avec le err_num
        exit(self._err_num)

    def log_min(self, *args, sep=' ', end='\n') -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        # self._log.log_level(args, sep, end, level='min')
        self._log.log_min(args, sep, end)

    def log_moy(self, *args, sep=' ', end='\n') -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        self._log.log_moy(args, sep, end)
        # self._log.log_level(args, sep, end, level='moy')

    def log_max(self, *args, sep=' ', end='\n') -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        # self._log.log_level(args, sep, end, level='max')
        self._log.log_max(args, sep, end)

    def log_err(self) -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        err = self._log.log_err()
        self._err_num = int(err[0])
        self._err_msg = err[1]

    def log_remove(self) -> None:
        self._log.log_remove()

    def log_memory(self, with_cout: bool = False) -> None:
        if self._log is None:
            self._log = llog.CtxLog(self._cmd_name)
        self._log.memory_uss(with_cout)


ctx = Context()
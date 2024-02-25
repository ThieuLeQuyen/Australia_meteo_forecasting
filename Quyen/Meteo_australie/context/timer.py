""" Module de gestion d'un timer pour les temps de calcul """

import io
import time

__all__ = ['current_nanosecond', 'current_microsecond', 'CtxTimer']

MICROSECOND_NANO: int = 1000
MILLISECOND_NANO: int = MICROSECOND_NANO * 1000
SECOND_NANO: int = MILLISECOND_NANO * 1000
MINUTE_NANO: int = SECOND_NANO * 60
HOUR_NANO: int = MINUTE_NANO * 60


def current_nanosecond() -> int:
    """
        Retourne le temps passé en nanosecond
        return int
    """
    return time.time_ns()


def current_microsecond() -> int:
    """
        Retourne le temps passé en microsecond
        return int
    """
    return int(time.time_ns() / MICROSECOND_NANO)


class CtxTimer:
    """
        Classe de getsion de timer
    """
    _dt_start: float
    _t_start: int
    _t_step: int

    def __init__(self) -> None:
        """
            Initialisation de l'objet de classe CtxTime
        """
        self.start()

    def start(self) -> int:
        """
            Initialise le timestamp de depart et d'étape et retourne le timestamp de depart.
            :return: int
        """
        self._dt_start = time.time()
        self._t_start = time.time_ns()
        self._t_step = self._t_start
        return self._t_start

    def step(self) -> int:
        """
            Retourne la différence entre le timestamp d'étape et le timestamp courant.
            :return: int
        """
        current = current_nanosecond()
        diff = current - self._t_step
        self._t_step = current
        return diff

    def stop(self) -> int:
        """
            Retourne la différence entre le timestamp de depart et le timestamp courant.
            :return: int
        """
        diff = current_nanosecond() - self._t_start
        return max(diff, 0)

    @staticmethod
    def to_string(diff_nanosecond: int) -> str:
        time_parts = [
            (HOUR_NANO, "h"),
            (MINUTE_NANO, "mn"),
            (SECOND_NANO, "s"),
            (MILLISECOND_NANO, "ms"),
            (MICROSECOND_NANO, "microsecond"),
        ]

        result = []
        for duration, label in time_parts:
            if diff_nanosecond >= duration:
                value, diff_nanosecond = divmod(diff_nanosecond, duration)
                result.append(f"{value} {label}")
        return " ".join(result) or "0 nanosecond"

    def start_to_string(self) -> str:
        """
            Convertit le temps départ en seconde en chaine de caracteres.
            :return: str
        """
        return time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(self._dt_start))

    def step_to_string(self) -> str:
        """
            Convertit un temps d'intervalle d'étape en nanosecondes en chaine de caracteres.
            :return: str
        """
        return self.to_string(self.step())

    def stop_to_string(self) -> str:
        """
            Convertit le temps de traitement en nanosecondes en chaine de caracteres.
            :return: str
        """
        return self.to_string(self.stop())


if __name__ == '__main__':
    t = CtxTimer()
    print(t.start_to_string())
    time.sleep(10)
    print(t.step_to_string())
    time.sleep(0.2)
    print(t.step_to_string())
    time.sleep(0.3)
    print(t.step_to_string())
    time.sleep(0.4)
    print(t.stop_to_string())
    # help(CtxTime)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Vertical bar plots
def bar_plot(df: pd.DataFrame, x_name: str, y_name: str,
             fig_size=(12, 8),
             title: str = "", x_label: str = "", y_label: str = "",
             xticks_rotation: int = 0):
    fig = plt.figure(figsize=fig_size)

    p = sns.barplot(data=df, x=x_name, y=y_name)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=xticks_rotation)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)

    # Annotate
    for bar in p.patches:
        p.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='center', size=12, xytext=(0, 8),
                   textcoords='offset points'
                   )

    plt.show()
    return fig


# Horizontal bar plots
def horizontal_bar_plot(df: pd.DataFrame, x_name: str, y_name: str,
                        fig_size=(8, 12),
                        title: str = "", x_label: str = "", y_label: str = "",
                        order=None, with_annotate: bool = True):
    """

    :param df:
    :param x_name:
    :param y_name:
    :param fig_size:
    :param title:
    :param x_label:
    :param y_label:
    :param order:
    :param with_annotate:
    :return:
    """
    fig = plt.figure(figsize=fig_size)

    if order is None:
        p = sns.barplot(data=df, x=x_name, y=y_name)
    else:
        p = sns.barplot(data=df, x=x_name, y=y_name, order=order)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)

    if with_annotate:
        # Annotate
        for bar in p.patches:
            p.annotate(format(bar.get_width(), '.2f'),
                       (bar.get_width() + 1, bar.get_y() + bar.get_height()),
                       ha='center', va='center', size=12, xytext=(0, 8),
                       textcoords='offset points'
                       )
    plt.tight_layout()
    # plt.show()
    return fig


def number_of_nan(ser: pd.Series, in_percentage: bool = True):
    """
    Calculer le nombre de valeurs manquantes d'une séries pandas
    Args:
        ser: une séries pandas de données
        in_percentage (bool): True (par défaut) si on veut retourner le pourcentage des valeurs manquantes
            False si on veut retourner le nombre des valeurs manquantes
    """
    if not in_percentage:
        return ser.isna().sum()
    else:
        return np.round(100 * (ser.isna().sum() / len(ser)), 2)


def _check_argument(param, options, value):
    """Sort une exception si la valeur du parametre n'est pas dans les options."""
    if value not in options:
        raise ValueError(f"`{param}` must be one of {options}, but {repr(value)} was passed.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from src.utils import horizontal_bar_plot


class Visualisation:
    """
    Cette classe visualise les données d'un dataframe
    """

    def __init__(self, df_data: pd.DataFrame):
        self.df_data = df_data
        self.list_of_numrical_variables = list(df_data.select_dtypes(include=np.number).columns)

    @staticmethod
    def _check_subplots(nrows: int, ncols: int, list_col: list) -> None:
        if nrows * ncols < len(list_col):
            raise Exception("le produit de nrows et ncols doit être supérieur au nombre d'élément de list_col")

    def barh_value_counts(self, list_col: list,
                          nrows: int = None, ncols: int = None,
                          fig_size=(10, 14),
                          with_annotate: bool = True):
        """

        :param list_col:
        :param nrows:
        :param ncols:
        :param fig_size:
        :return:
        """
        if len(list_col) == 1:
            var = list_col[0]
            counts = self.df_data[var].value_counts()
            df_counts = pd.DataFrame({'modality': counts.index, 'count': counts.values})

            fig = horizontal_bar_plot(df_counts, y_name="modality", x_name="count",
                                      fig_size=fig_size,
                                      title=f"Distribution de la variable {var}",
                                      y_label=var, with_annotate=with_annotate)
            return fig

        if (nrows is None) or (ncols is None):
            nrows = len(list_col)
            ncols = 1

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
        for i, var in enumerate(list_col):
            counts = self.df_data[var].value_counts()
            df_counts = pd.DataFrame({'modality': counts.index, 'count': counts.values})
            p = sns.barplot(data=df_counts, x='count', y='modality', ax=ax[i])
            ax[i].set_title(var)
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
            if with_annotate:
                # Annotate
                for bar in p.patches:
                    p.annotate(format(bar.get_width(), '.2f'),
                               (bar.get_width() + 1, bar.get_y() + bar.get_height()),
                               ha='center', va='center', size=12, xytext=(0, 8),
                               textcoords='offset points'
                               )

        return fig

    def histogram_missing_by_row(self, bins: list = None, fig_size=(14, 14),
                                 xlabel: str = "% des valeurs manquantes par ligne",
                                 title="Histogramme des données manquantes par observation"):
        pct_missing_by_row = (self.df_data.isna().mean(axis=1) * 100).to_frame()
        pct_missing_by_row.columns = ["pct_missing"]

        fig, ax = plt.subplots(1, 1)

        if bins is not None:
            ax.hist(pct_missing_by_row, rwidth=0.8, bins=bins)
        else:
            ax.hist(pct_missing_by_row, rwidth=0.8)

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        for rect in ax.patches:
            label = f"{round(100 * rect.get_height() / self.df_data.shape[0], 2)} %"
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 5,
                    label, ha="center", va="bottom")
        return fig

    def barh_missing_by_column(self, fig_size=(14, 14), list_col: list = None,
                               title: str = "% de données manquantes", ascending: bool = True):
        """Visualise le pourcentage de données manquantes par colonne"""

        if list_col is None:
            list_col = self.df_data.columns

        pct_missing_by_col = (self.df_data[list_col].isna().mean() * 100).sort_values(ascending=ascending).to_frame()
        pct_missing_by_col.columns = ["pct_missing"]
        pct_missing_by_col["variable"] = pct_missing_by_col.index

        fig = horizontal_bar_plot(pct_missing_by_col, y_name="variable", x_name="pct_missing",
                                  fig_size=fig_size,
                                  title=title,
                                  y_label="Variable", x_label="%")
        return fig

    def boxplot(self, nrows: int, ncols: int, list_col: list = None,
                fig_size=(20, 20), title: str = "Boxplots des variables quantatives"):
        """Visualise des boxplots pour les colonnes numériques"""

        if list_col is None:
            list_col = self.list_of_numrical_variables

        self._check_subplots(nrows, ncols, list_col)

        fig_box, axes = plt.subplots(figsize=fig_size, nrows=nrows, ncols=ncols)
        axes = axes.reshape(-1)

        for i, col in enumerate(list_col):
            sns.boxplot(self.df_data[col], ax=axes[i])
            axes[i].title.set_text(col)

        plt.tight_layout(pad=6.0)
        fig_box.suptitle(title, fontsize=25)
        return fig_box

    def histogram(self, nrows: int, ncols: int, list_col: list = None,
                  fig_size=(20, 20), title: str = "Boxplots des variables quantatives"):
        """Visualise des histogrammes pour les colonnes numériques"""

        if list_col is None:
            list_col = self.list_of_numrical_variables

        self._check_subplots(nrows, ncols, list_col)

        fig_hist, axes = plt.subplots(figsize=fig_size, nrows=nrows, ncols=ncols)
        axes = axes.reshape(-1)

        for i, col in enumerate(list_col):
            sns.histplot(self.df_data[col].fillna(np.nan), ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].title.set_text(col)

        plt.tight_layout(pad=5.0)
        fig_hist.suptitle(title, fontsize=25)
        return fig_hist

    def heatmap_correlation(self, list_col: list = None,
                            fig_size=(20, 20), title: str = "Corrélation entre les variables numériques"):
        """Affiche une heatmap des corrélations entre les colonnes numériques"""

        if list_col is None:
            list_col = self.list_of_numrical_variables

        cmap = sns.diverging_palette(260, 20, as_cmap=True)

        fig_corr, ax = plt.subplots(figsize=fig_size)
        corr_mat = self.df_data[list_col].corr()
        mask = np.triu(np.ones_like(corr_mat))

        sns.heatmap(corr_mat,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    ax=ax)
        ax.set_title(title, fontsize=20)
        return fig_corr

    def matrix_of_nullity(self, fig_size=(25, 14)):
        """Affiche une matrice de nullité pour visualiser les données manquantes"""

        ax = msno.matrix(self.df_data, figsize=fig_size)
        fig = ax.get_figure()
        plt.title("Matrice de nullité", fontsize=20)
        plt.tight_layout()
        return fig

    def heatmap_of_nullity(self, fig_size=(20, 14)):
        """Affiche une heatmap pour visualiser les corrélations de données manquantes"""

        ax = msno.heatmap(self.df_data, figsize=fig_size)
        fig = ax.get_figure()
        plt.title("Matrice de corrélation de nullité", fontsize=20)
        plt.tight_layout()
        return fig

    def dendrogram(self, fig_size=(20, 14)):
        """Affiche un dendrogramme pour visualiser les schémas de données manquantes"""

        ax = msno.dendrogram(self.df_data, figsize=fig_size)
        fig = ax.get_figure()
        plt.title("Dendrogram de nullité", fontsize=20)
        plt.tight_layout()
        return fig

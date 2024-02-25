# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer

from src.utils import _check_argument


class Preprocessing:
    """

    """
    def __init__(self, df_data: pd.DataFrame):
        self.df_data = df_data.copy()
        self.list_categorial_variables = list(self.df_data.select_dtypes(include=["object", "category"]).columns)
        self.list_numeric_variables = list(self.df_data.select_dtypes(include=["number"]).columns)

    def get_dummies(self, list_categorial_variables: list = None):
        """
        Encoder une liste des variables catégorielles du dataframe
        :param list_categorial_variables:
        :return: un dataframe
        """
        if list_categorial_variables is None:
            # encoder toutes les variables catégorielles du dataframe
            list_categorial_variables = self.list_categorial_variables
        if len(list_categorial_variables) == 0:
            print("Il n'y a pas de variable catégorielle dans le dataframe")
            return self.df_data
        return pd.get_dummies(self.df_data, columns=list_categorial_variables)

    def normalize(self):
        """
        Normaliser le dataframe.
        Si le dataframe contient des variables catégorielles, il faut les encoder avant de normaliser
        :return: un dataframe avec les colonnes des variables numériques données qui sont normalisées
        """
        if len(self.list_categorial_variables) > 0:
            df_scaled = self.get_dummies()
        else:
            df_scaled = self.df_data.copy()

        scaler = StandardScaler()
        df_res = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)
        return {'df_scaled': df_res,
                'scaler': scaler}


class HandlingMissingValues(Preprocessing):
    """
    Traiter des valeurs manquantes du dataframe
    """
    def __init__(self, df_data_: pd.DataFrame):
        super().__init__(df_data_)

    def delete_rows(self, subset: list = None, pct_missing: float = None):
        """
        Supprimer des lignes contenant des valeurs manquantes selon des contraintes
        :param subset: une liste des colonnes à prendre en compte
        :param pct_missing: le seuil de taux maximal de valeurs manquantes pour une ligne
        :return:
        """
        if pct_missing is None:
            if subset is None:
                self.df_data.dropna(axis=0, how='any', inplace=True)
            else:
                # contrôle si subset appartient aux colonnes du dataframe
                if set(subset).issubset(self.df_data.columns):
                    self.df_data.dropna(axis=0, how='any', subset=subset, inplace=True)
                else:
                    print("Les colonnes spécifiées ne sont pas présentes dans le dataframe")
        else:
            if subset is None:
                df_pct_missing = pd.DataFrame(self.df_data.isna().mean(axis=1) * 100, columns=["pct_missing"])
            else:
                df_pct_missing = pd.DataFrame(self.df_data[subset].isna().mean(axis=1) * 100, columns=["pct_missing"])

            indices_to_keep = df_pct_missing[df_pct_missing['pct_missing'] < pct_missing].index
            self.df_data = self.df_data.loc[indices_to_keep, :]

    def mean_imputer(self, list_numerical_variables: list = None):
        """

        :param list_numerical_variables:
        :return:
        """
        if list_numerical_variables is None:
            # imputer des valeurs manquantes sur toutes les variables numériques
            list_numerical_variables = list(self.df_data.select_dtypes(include="number").columns)

        imp_mean = SimpleImputer(strategy="mean")
        df_imputed = self.df_data.copy()
        df_imputed[list_numerical_variables] = imp_mean.fit_transform(self.df_data[list_numerical_variables])
        return df_imputed

    def median_imputer(self, list_numerical_variables: list = None):
        """

        :param list_numerical_variables:
        :return:
        """
        if list_numerical_variables is None:
            # imputer des valeurs manquantes sur toutes les variables numériques
            list_numerical_variables = list(self.df_data.select_dtypes(include="number").columns)

        imp_median = SimpleImputer(strategy="median")
        df_imputed = self.df_data.copy()
        df_imputed[list_numerical_variables] = imp_median.fit_transform(self.df_data[list_numerical_variables])
        return df_imputed

    def knn_imputer(self, n_neighbors: int, weights='uniform', normaliser_features: bool = True):
        """

        :param weights:
        :param normaliser_features: True (par défaut) s'il faut scaler les features avant d'imputer les valeurs manquantes
        :param n_neighbors:
        :return:
        """
        _check_argument('weights', ['uniform', "distance"], weights)

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        if normaliser_features:
            print("start")
            res_scaled = self.normalize()
            df_scaled = res_scaled['df_scaled']
            scaler = res_scaled['scaler']
            print("fini scale")
            df_imputed = imputer.fit_transform(df_scaled)
            print("fini fit_transform")
            df_imputed = pd.DataFrame(df_imputed, columns=df_scaled.columns)
            print("fini dataframe")
            # Revenir à l'échelle d'origine pour les variables numériques
            df_imputed_inverse = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df_scaled.columns)
            return {'df_imputed': df_imputed,
                    'df_imputed_inverse': df_imputed_inverse}
        else:
            df_imputed = imputer.fit_transform(self.df_data)
            df_imputed = pd.DataFrame(df_imputed, columns=self.df_data.columns)
            return {'df_imputed': df_imputed,
                    'df_imputed_inverse': df_imputed}


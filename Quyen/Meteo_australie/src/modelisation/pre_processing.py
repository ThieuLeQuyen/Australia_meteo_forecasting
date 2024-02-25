# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""
import time

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
import os

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from context.context import ctx


def tweak_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({'Yes': 1, 'No': 0})


class TweakTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, y_col=None):
        self.y_col = y_col

    def fit(self, X, y):
        # ne fait rien
        return self

    def transform(self, X):
        return tweak_data(X)


class ColumnDrop(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_drop):
        self.column_to_drop = column_to_drop

    def fit(self, X, y):
        # ne fait rien
        return self

    def transform(self, X):
        # Suppresion de la colonne
        return X.drop(self.column_to_drop, axis=1)


class RowDrop(BaseEstimator, TransformerMixin):
    def __init__(self, pct_missing):
        self.pct_missing = pct_missing  # en %

    def fit(self, X, y):
        # ne fait rien
        return self

    def transform(self, X):
        df_pct_missing = pd.DataFrame(X.isna().mean(axis=1) * 100, columns=["pct_missing"])
        indices_to_keep = df_pct_missing[df_pct_missing['pct_missing'] < self.pct_missing].index
        return X.loc[indices_to_keep, :]


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_dummies):
        self.cols_to_dummies = cols_to_dummies

    def fit(self, X, y):
        # ne fait rien
        return self

    def transform(self, X):
        return pd.get_dummies(X, columns=self.cols_to_dummies)


class Normalization(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_normalize: list = None):
        self.cols_to_normalize = cols_to_normalize
        self.scaler = StandardScaler()

    def fit(self, X, y):
        if self.cols_to_normalize:
            self.scaler.fit(X[self.cols_to_normalize])
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X):
        if self.cols_to_normalize is None:
            # normalize toutes les colonnes
            return pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        else:
            X[self.cols_to_normalize] = self.scaler.transform(X[self.cols_to_normalize])
            return X


class InverseScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(self.scaler.inverse_transform(X), columns=X.columns)


class KNNImputation(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_impute: list = None, n_neighbors: int = 2, weights: str = "uniform"):
        self.cols_to_impute = cols_to_impute
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        # ne fait rien
        return self

    def transform(self, X):
        imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        if self.cols_to_impute is None:
            # normalize toutes les colonnes
            return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        else:
            return pd.DataFrame(imputer.fit_transform(X[self.cols_to_impute]),
                                columns=X[self.cols_to_impute].columns)


class PreProcessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_raw_X_y(self, y_col, drop_nan: bool = False):
        ctx.log_min(" --- PreProcessing.get_raw_X_y ---")
        if drop_nan:
            raw = self.df.dropna(axis=0, how='any')
        else:
            raw = self.df.dropna(axis=0, how='any', subset=[y_col])
        return raw.drop(y_col, axis=1), raw[y_col]

    def get_train_test(self, y_col, cols_to_drop: list = None,
                       drop_all_nan: bool = False,
                       cols_to_dummies: list = None,
                       pct_missing_row: float = None,
                       n_neighbors: int = 2, weights="uniform",
                       return_original_data: bool = True,
                       test_size: float = 0.2, random_state: int = 42, is_stratify: bool = True):

        ctx.log_min(" --- PreProcessing.get_train_test() ---")
        ctx.log_min("drop_all_nan = ", drop_all_nan)
        ctx.log_min("ycol = ", y_col)
        if drop_all_nan:
            ctx.log_min("enlever toutes les lignes contenant des nan")
            X, y = self.get_raw_X_y(y_col=y_col, drop_nan=True)
        else:
            ctx.log_min("récupérer X et y")
            X, y = self.get_raw_X_y(y_col=y_col)

        ctx.log_min("X.shape = ", X.shape)

        if is_stratify:
            X_raw_train, X_raw_test, y_train, y_test = train_test_split(X, y,
                                                                        test_size=test_size,
                                                                        random_state=random_state,
                                                                        stratify=y)
        else:
            X_raw_train, X_raw_test, y_train, y_test = train_test_split(X, y,
                                                                        test_size=test_size,
                                                                        random_state=random_state)

        ctx.log_min("X_raw_train.shape = ", X_raw_train.shape)
        ctx.log_min("X_raw_test.shape = ", X_raw_test.shape)

        # Instanciation des transformeurs de la pipeline
        steps = []

        ctx.log_min("TweakTransformer")
        tweak = TweakTransformer()
        steps.append(('tweak_data', tweak))

        if cols_to_drop is not None:
            ctx.log_min("ColumnDrop")
            col_drop = ColumnDrop(column_to_drop=cols_to_drop)
            steps.append(('col_drop', col_drop))

        if pct_missing_row is not None:
            ctx.log_min("RowDrop")
            row_drop = RowDrop(pct_missing=pct_missing_row)
            steps.append(('row_drop', row_drop))

        if cols_to_dummies is not None:
            ctx.log_min("GetDummies")
            get_dummies = GetDummies(cols_to_dummies=cols_to_dummies)
            steps.append(('get_dummies', get_dummies))

        ctx.log_min("Normalization")
        scaler = Normalization()
        steps.append(('scaler', scaler))

        ctx.log_min("KNNImputation")
        knn_imputer = KNNImputation(n_neighbors=n_neighbors, weights=weights)
        steps.append(('knn_imputer', knn_imputer))

        if return_original_data:
            ctx.log_min("Inverse scaler")
            inverse_scaler = InverseScaler(scaler=scaler.scaler)
            steps.append(('inverse_scaler', inverse_scaler))

        ctx.log_min("Pipeline - start")
        t1 = time.time()
        prep_pipe = Pipeline(steps=steps)

        X_train = prep_pipe.fit_transform(X_raw_train, y_train)
        X_test = prep_pipe.transform(X_raw_test)
        t2 = time.time()

        ctx.log_min(f"Pipeline - end - temps de calcul = {round(t2 - t1, 2)} seconds")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    ctx.cmd_start()
    fichier_data = "../../data/weatherAUS.csv"

    df_data = pd.read_csv(fichier_data)
    df_data = df_data.sample(10_000)

    var_cat = ['WindDir9am', 'WindDir3pm', 'WindGustDir']
    t1 = time.time()
    prep = PreProcessing(df_data)
    X_train, X_test, y_train, y_test = prep.get_train_test(y_col="RainTomorrow",
                                                           cols_to_drop=["Date", "Location"],
                                                           drop_all_nan=False,
                                                           cols_to_dummies=var_cat,
                                                           pct_missing_row=None,
                                                           n_neighbors=2, weights="uniform",
                                                           test_size=0.2,
                                                           random_state=42, is_stratify=False)

    t2 = time.time()
    print(t2 - t1)

    stump_dt = tree.DecisionTreeClassifier(max_depth=1)

    stump_dt.fit(X_train, y_train)
    stump_dt.score(X_test, y_test)

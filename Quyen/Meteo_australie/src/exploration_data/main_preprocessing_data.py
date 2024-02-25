# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""

import time

import pandas as pd

from src.exploration_data.preprocessing_data import HandlingMissingValues

fichier_data = "./data/weatherAUS.csv"

df_data = pd.read_csv(fichier_data)


def tweak_data(df: pd.DataFrame):
    """
    :param df:
    :return:
    """
    return (df
            .replace({'Yes': 1, 'No': 0})
            .set_index(pd.to_datetime(df.Date))
            .drop('Date', axis=1)
            )


df_cleaned_data = tweak_data(df_data)

hmv = HandlingMissingValues(df_cleaned_data)

# =============================================================================================
# Suppression par ligne
# =============================================================================================
# Enlever les lignes où la colonne "RainTomorrow" n'a pas de donnée
hmv.delete_rows(subset=["RainTomorrow"])

# Enlever les lignes où il y a plus de 50% de données manquantes
hmv.delete_rows(pct_missing=50)

# =============================================================================================
# Imputation by Mean
# =============================================================================================
var_num = list(df_cleaned_data.select_dtypes(include="number").columns)
var_num.remove("RainToday")
var_num.remove("RainTomorrow")

df_imputed_by_mean = hmv.mean_imputer(var_num)

fichier_out = f"./output/df_imputed_by_mean.csv"
df_imputed_by_mean.to_csv(fichier_out, sep=";", index=False)

# =============================================================================================
# Imputation by Median
# =============================================================================================

df_imputed_by_median = hmv.median_imputer(var_num)

fichier_out = f"./output/df_imputed_by_median.csv"
df_imputed_by_median.to_csv(fichier_out, sep=";", index=False)

# =============================================================================================
# KNN Imputaion
# =============================================================================================
k = 2

for k in [2]:
    print(k)
    t1 = time.time()
    res_knn = hmv.knn_imputer(n_neighbors=k, weights='uniform', normaliser_features=True)

    df_imputed = res_knn['df_imputed']
    fichier_out = f"./output/df_imputed_knn_{k}.csv"
    df_imputed.to_csv(fichier_out, sep=";", index=False)

    df_imputed_inverse = res_knn['df_imputed_inverse']
    fichier_out = f"./output/df_imputed_inverse_{k}.csv"
    df_imputed_inverse.to_csv(fichier_out, sep=";", index=False)

    t2 = time.time()
    print("temps = ", t2 - t1)




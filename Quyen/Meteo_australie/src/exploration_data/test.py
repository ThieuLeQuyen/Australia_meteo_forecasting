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

df_data_full = pd.read_csv(fichier_data)

# =============================================================================================
# Suppression par ligne
# =============================================================================================
# Enlever les lignes où la colonne "RainTomorrow" n'a pas de donnée
df_data = df_data_full.dropna(axis=0, how='any', subset=['RainTomorrow'])

# Enlever les lignes où il y a plus de 50% de données manquantes
pct_missing_by_row = pd.DataFrame(df_data.isna().mean(axis=1) * 100, columns=["pct_missing"])
threshold_missing_by_obs = 50  # à décider
df_data = df_data.loc[pct_missing_by_row[pct_missing_by_row["pct_missing"] < threshold_missing_by_obs].index, :]

df_data = df_data.replace({'Yes': 1, 'No': 0})

df_taux = []
for i, loca in enumerate(df_data["Location"].unique()):
    print(i, " - ", loca)
    df = df_data.loc[df_data["Location"] == loca]
    taux = 100 - 100 * df["RainTomorrow"].sum() / df.shape[0]
    df_taux.append([loca, taux])

df_res = pd.DataFrame(df_taux)

100 - df_res.mean()
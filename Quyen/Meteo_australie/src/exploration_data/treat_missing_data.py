import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer

print(os.getcwd())
fichier_data = "./data/weatherAUS.csv"

df_data = pd.read_csv(fichier_data)


def tweak_data_bis(df: pd.DataFrame):
    return (df
            .replace({'Yes': 1, 'No': 0})
            .set_index(pd.to_datetime(df.Date))
            .drop('Date', axis=1)
            )


def tweak_data(df: pd.DataFrame):
    return (df
            .replace({'Yes': 1, 'No': 0})
            .drop('Date', axis=1)
            )


df_cleaned_data = tweak_data(df_data)

# =============================================================================================
# Suppression par ligne
# =============================================================================================
# Enlever les lignes où la colonne "RainTomorrow" n'a pas de donnée
df_cleaned_data = df_cleaned_data.dropna(axis=0, how='any', subset=['RainTomorrow'])

# Enlever les lignes où il y a plus de 50% de données manquantes
pct_missing_by_row = pd.DataFrame(df_cleaned_data.isna().mean(axis=1) * 100, columns=["pct_missing"])
threshold_missing_by_obs = 50  # à décider
df_cleaned_data = df_cleaned_data.loc[
                  pct_missing_by_row[pct_missing_by_row["pct_missing"] < threshold_missing_by_obs].index, :]

# =============================================================================================
# Imputation by Mean
# =============================================================================================
var_num = list(df_cleaned_data.select_dtypes(include="number").columns)
var_num.remove("RainToday")
var_num.remove("RainTomorrow")

imp_mean = SimpleImputer(strategy="mean")
df_imputed_by_mean = df_cleaned_data.copy()
df_imputed_by_mean[var_num] = imp_mean.fit_transform(df_imputed_by_mean[var_num])

fichier_out = f"./output/df_imputed_by_mean.csv"
df_imputed_by_mean.to_csv(fichier_out, sep=";", index=False)

# =============================================================================================
# Imputation by Median
# =============================================================================================
imp_median = SimpleImputer(strategy="median")
df_imputed_by_median = df_cleaned_data.copy()
df_imputed_by_median[var_num] = imp_median.fit_transform((df_imputed_by_median[var_num]))
fichier_out = f"./output/df_imputed_by_median.csv"
df_imputed_by_median.to_csv(fichier_out, sep=";", index=False)

# =============================================================================================
# KNN Imputaion
# =============================================================================================
var_cat = ['WindDir9am', 'WindDir3pm', 'WindGustDir']

# Etape 1: Encodage One-Hot des variables catégorielles
df_dummies_data = pd.get_dummies(df_cleaned_data, columns=var_cat)

# Essaie avec une petite partie des données (juste pour voir le temps de calcul)
# df_cleaned_data = df_cleaned_data.loc[df_cleaned_data['Location'] == "Townsville", :]

df_dummies_data = df_dummies_data.drop("Location", axis=1)

# Étape 2: Normalisation des variables numériques
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_dummies_data), columns=df_dummies_data.columns)

# Étape 3: Imputation KNN
# Utilisation de fancyimpute pour remplir les valeurs manquantes

k = 3

for k in range(2, 6):
    print(k)
    t1 = time.time()
    imputer = KNNImputer(n_neighbors=k, weights='uniform')
    df_imputed = imputer.fit_transform(df_scaled)
    df_imputed = pd.DataFrame(df_imputed, columns=df_scaled.columns)

    # Si vous voulez revenir à l'échelle d'origine pour les variables numériques
    df_imputed_inverse = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df_scaled.columns)

    fichier_out = f"./output/df_imputed_knn_{k}.csv"
    df_imputed.to_csv(fichier_out, sep=";", index=False)

    fichier_out = f"./output/df_imputed_inverse_{k}.csv"
    df_imputed_inverse.to_csv(fichier_out, sep=";", index=False)

    t2 = time.time()
    print("temps = ", t2 - t1)




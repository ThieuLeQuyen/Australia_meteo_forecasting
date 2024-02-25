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
            )


df_cleaned_data = tweak_data(df_data)

# =============================================================================================
# Suppression par ligne
# =============================================================================================
# Enlever les lignes où la colonne "RainTomorrow" n'a pas de donnée
df_cleaned_data = df_cleaned_data.dropna(axis=0, how='any', subset=['RainTomorrow'])

# Enlever les lignes où il y a plus de 50% de données manquantes
pct_missing_by_row = pd.DataFrame(df_cleaned_data.drop('Date', axis=1).isna().mean(axis=1) * 100, columns=["pct_missing"])
threshold_missing_by_obs = 50  # à décider
df_cleaned_data = df_cleaned_data.loc[
                  pct_missing_by_row[pct_missing_by_row["pct_missing"] < threshold_missing_by_obs].index, :]

# =============================================================================================
# Charger les données imputées
# =============================================================================================

# Imputation by mean
fichier_imputed = f"./output/df_imputed_by_mean.csv"
df_imputed_by_mean = pd.read_csv(fichier_imputed, sep=';')

# Imputation by median
fichier_imputed = f"./output/df_imputed_by_median.csv"
df_imputed_by_median = pd.read_csv(fichier_imputed, sep=';')

# KNN Imputation
k = 2
fichier_imputed = f"./output/df_imputed_inverse_{k}.csv"
df_imputed_inverse_2 = pd.read_csv(fichier_imputed, sep=';')

k = 3
fichier_imputed = f"./output/df_imputed_inverse_{k}.csv"
df_imputed_inverse_3 = pd.read_csv(fichier_imputed, sep=';')

k = 4
fichier_imputed = f"./output/df_imputed_inverse_{k}.csv"
df_imputed_inverse_4 = pd.read_csv(fichier_imputed, sep=';')

# =============================================================================================
# Déterminer les variables numériques à comparer
# =============================================================================================

pct_missing_by_col = ((df_cleaned_data.isna().sum() * 100) / df_cleaned_data.shape[0]).sort_values(
    ascending=False).to_frame()
quantitative_features = list(pct_missing_by_col.index)
list_var_exclu = ["RainToday", "RainTomorrow", "Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
for var in list_var_exclu:
    quantitative_features.remove(var)

if "Date" in quantitative_features:
    quantitative_features.remove("Date")
# =============================================================================================
# Comparaison de le boxplot avant et après imputation
# =============================================================================================

fig_box, axes = plt.subplots(figsize=(20, 20), nrows=4, ncols=4)
axes = axes.reshape(-1)

for i, col in enumerate(quantitative_features):
    df = pd.DataFrame({"Avant": df_cleaned_data[col],
                       "k=2": df_imputed_inverse_2[col],
                       "k=3": df_imputed_inverse_3[col],
                       "k=4": df_imputed_inverse_4[col]})
    df = df.loc[pd.isnull(df["Avant"]), ["k=2", "k=3", "k=4"]]
    sns.boxplot(df, ax=axes[i])

    title = f"{col} - {df.shape[0]} valeurs"
    axes[i].title.set_text(title)
    axes[i].legend()

plt.tight_layout(pad=6.0)
fig_box.suptitle("Boxplots des valeurs ajoutées par KNN imputation",
                 fontsize=25)
plt.show()

fichier_out = "./output/Boxplot_new_values_KNN_imputation.png"
fig_box.savefig(fichier_out)

# =============================================================================================
# Comparaison de la densité avant et après imputation KNN
# =============================================================================================

title = f'Densité avant et après KNN imputation'
fig, axes = plt.subplots(figsize=(20, 20), nrows=4, ncols=4)
axes = axes.reshape(-1)
sns.set(style="darkgrid")
for i, col in enumerate(quantitative_features):
    sns.kdeplot(df_cleaned_data[col], linewidth=4, fill=False, color='black', ax=axes[i], label="Avant")
    sns.kdeplot(df_imputed_inverse_2[col], linewidth=3, fill=False, color='green', ax=axes[i], label=f"k = {2}")
    sns.kdeplot(df_imputed_inverse_3[col], linewidth=2, fill=False, color='red', ax=axes[i], label=f"k = {3}")
    sns.kdeplot(df_imputed_inverse_4[col], linewidth=1, fill=False, color='blue', ax=axes[i], label=f"k = {4}")
    axes[i].title.set_text(col)
    axes[i].legend()
plt.tight_layout(pad=6.0)
fig.suptitle(title, fontsize=25)
# plt.show()

fichier_out = "./output/Density_before_after_KNN_imputation.png"
fig.savefig(fichier_out)

# =============================================================================================
# Comparaison de le boxplot avant et après imputation KNN
# =============================================================================================

fig_box, axes = plt.subplots(figsize=(20, 20), nrows=4, ncols=4)
axes = axes.reshape(-1)

for i, col in enumerate(quantitative_features):
    df = pd.DataFrame({"Avant": df_cleaned_data[col],
                       "k=2": df_imputed_inverse_2[col],
                       "k=3": df_imputed_inverse_3[col],
                       "k=4": df_imputed_inverse_4[col]})
    sns.boxplot(df, ax=axes[i])

    axes[i].title.set_text(col)
    axes[i].legend()

plt.tight_layout(pad=6.0)
fig_box.suptitle("Boxplots avant et après KNN imputation",
                 fontsize=25)
plt.show()

fichier_out = "./output/Boxplot_before_after_KNN_imputation.png"
fig_box.savefig(fichier_out)

# =============================================================================================
# Comparaison de la densité avant et après imputation par mean, median et KNN
# =============================================================================================

title = f'Densité avant et après imputation par moyenne/médiane'
fig, axes = plt.subplots(figsize=(20, 20), nrows=4, ncols=4)
axes = axes.reshape(-1)
sns.set(style="darkgrid")
for i, col in enumerate(quantitative_features):
    sns.kdeplot(df_cleaned_data[col], linewidth=4, fill=False, color='black', ax=axes[i], label="Avant")
    sns.kdeplot(df_imputed_by_mean[col], linewidth=2, fill=False, color='red', ax=axes[i], label="mean imputation")
    sns.kdeplot(df_imputed_by_median[col], linewidth=1, fill=False, color='blue', ax=axes[i], label=f"median imputaion")
    axes[i].title.set_text(col)
    axes[i].legend()
plt.tight_layout(pad=6.0)
fig.suptitle(title, fontsize=25)
# plt.show()

fichier_out = "./output/Density_before_after_imputation_mean_median.png"
fig.savefig(fichier_out)

# =============================================================================================
# Comparaison de le boxplot avant et après imputation
# =============================================================================================

fig_box, axes = plt.subplots(figsize=(20, 20), nrows=4, ncols=4)
axes = axes.reshape(-1)

for i, col in enumerate(quantitative_features):
    df = pd.DataFrame({"Avant": df_cleaned_data[col],
                       "mean imputation": df_imputed_by_mean[col],
                       "median imputation": df_imputed_by_median[col]})
    sns.boxplot(df, ax=axes[i])

    axes[i].title.set_text(col)
    axes[i].legend()

plt.tight_layout(pad=6.0)
fig_box.suptitle("Boxplots avant et après imputation par moyenne/médiane",
                 fontsize=25)
plt.show()

fichier_out = "./output/Boxplot_before_after_imputation_mean_median.png"
fig_box.savefig(fichier_out)
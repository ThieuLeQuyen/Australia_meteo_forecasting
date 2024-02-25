# -*- coding: utf-8 -*-

import os

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump

print(os.getcwd())
fichier_data = "./data/weatherAUS.csv"

df_data = pd.read_csv(fichier_data)


def tweak_data(df: pd.DataFrame):
    return (df
            .replace({'Yes': 1, 'No': 0})
            .drop('Date', axis=1)
            )


df_cleaned_data = tweak_data(df_data)

print(df_cleaned_data.head())
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

fichier = './output/df_imputed_knn_2.csv'
print(fichier)

X = pd.read_csv(fichier, sep=";")
X = X.drop("RainTomorrow", axis=1)
y = (df_cleaned_data["RainTomorrow"]
     .astype(int)
     .replace({1: 'Yes', 0: 'No'}))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# =============================================================================================
# test un modèle
# =============================================================================================
print("start")
start = time.time()
params = {'C': 0.1, 'kernel': 'rbf', 'gamma': 0.1}
svm = SVC(**params)
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

end = time.time()
print("time = ", end - start)

# =============================================================================================
# hyperparamètres
# =============================================================================================

params = {'C': [0.1, 1, 5, 10],
          'kernel': ['rbf', 'linear', 'poly'],
          'gamma': [0.001, 0.1, 0.5]}

svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=params, scoring='accuracy')

grid_search.fit(X=pd.concat([X_train, X_test]), y=pd.concat([y_train, y_test]))
best_params = grid_search.best_params_

# Sauvegarder les résultats

dump(grid_search, "./output/modelisation/grid_search_svm.joblib")
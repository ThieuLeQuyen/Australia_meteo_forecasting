# -*- coding: utf-8 -*-

import os
import pandas as pd
import time
from hyperopt import fmin, tpe, hp, Trials
from joblib import dump
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

print(os.getcwd())
fichier_data = "./data/weatherAUS.csv"

df_data = pd.read_csv(fichier_data)


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

fichier = './output/df_imputed_knn_2.csv'

X = pd.read_csv(fichier, sep=";")
X = X.drop("RainTomorrow", axis=1)
y = (df_cleaned_data["RainTomorrow"]
     .astype(int)
     .replace({1: 'Yes', 0: 'No'}))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# =============================================================================================
# hyperopt pour déterminer les hyperparameters optimaux
# =============================================================================================

def objective_function(parameters):
    """
    Définir la fonction objective
    :param parameters:
    :return:
    """
    # Initiate RandomForestClassifier
    clf = ensemble.RandomForestClassifier(**parameters)

    # Calculer the mean cross-validation score using 5 folds
    score = cross_val_score(clf, X, y, cv=5).mean()
    return -score


# Définir l'espace des hyperparametres
search_space = {
    'n_estimators': hp.choice('n_estimators', range(10, 12)),
    'max_depth': hp.choice('max_depth', range(10, 12)),
}

# Trials object to store the results
trials = Trials()

start = time.time()
# Run optimization
best = fmin(fn=objective_function,
            space=search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=100)

end = time.time()
print("temps = ", end - start)

# Sauvegarder les résultats

dump(best, "./output/modelisation/best_random_forest.joblib")
dump(trials, "./output/modelisation/trials_random_forest.joblib")

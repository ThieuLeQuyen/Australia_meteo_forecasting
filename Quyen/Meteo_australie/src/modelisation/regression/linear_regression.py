# -*- coding: utf-8 -*-

"""
@created: 10/2023
@updated:
@author: quyen@marcaud.fr
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from scipy.stats import probplot
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from context.context import ctx

os.chdir("/Users/quyen/PycharmProjects/Meteo_australie")
print(os.getcwd())
model = "linear_regression"

k = 4
with_location = False

var_cible = "MaxTemp"

log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}_early_stopping"

ctx.cmd_start(file_log_name=log_file, verbose_level="min")

if with_location:
    path_output = f"./output/regression/knn_impute_{k}_with_location/{var_cible}/{model}"
    path_regression = f"./output/regression/knn_impute_{k}_with_location"
    data_file = f"./output/regression/data_pre_processing_with_location_knn_imputed_{k}_{var_cible}.joblib"
else:
    path_output = f"./output/regression/knn_impute_{k}_without_location/{var_cible}/{model}"
    path_regression = f"./output/regression/knn_impute_{k}"
    data_file = f"./output/regression/data_pre_processing_without_location_knn_imputed_{k}_{var_cible}.joblib"

os.makedirs(path_output, exist_ok=True)


def linear_regression(X_train, X_test, y_train, y_test, nom_model: str = None):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    coeffs = list(lr.coef_)
    coeffs.insert(0, lr.intercept_)

    feats = list(X_train.columns)
    feats.insert(0, 'intercept')

    df_coefs = pd.DataFrame({'valeur estimée': coeffs}, index=feats)

    print("Coefficients")
    print(df_coefs)
    r2_model = lr.score(X_train, y_train)
    scores_cvs = cross_val_score(lr, X_train, y_train)
    score_cv = scores_cvs.mean()

    print(r2_model)
    print(score_cv)

    # le score du modèle sur l'ensemble de test.
    r2_test = lr.score(X_test, y_test)
    print(r2_test)

    # nuage de points entre pred_test et y_test
    pred_test = lr.predict(X_test)
    y_min = y_test.min()
    y_max = y_test.max()

    fig = plt.figure()
    plt.scatter(pred_test, y_test)
    plt.plot([y_min, y_max], [y_min, y_max])
    plt.xlabel("pred_test")
    plt.ylabel("y_test")
    plt.title(f"{var_cible} - {nom_model} - R2 = {r2_test}")
    # plt.show()

    file_out = f"{path_output}/Nuage_de_points_ypred_ytest_{nom_model}.png"
    fig.savefig(file_out)

    # Les valeurs ajustées et les résidus du modèle -> afficher les résidus
    pred_train = lr.predict(X_train)
    residus = pred_train - y_train

    fig = plt.figure()
    plt.scatter(y_train, residus)
    plt.plot((y_train.min(), y_train.max()), (0, 0), lw=3, color='#0a5798')
    plt.xlabel("y_train")
    plt.ylabel("residus")
    plt.title(f"{var_cible} - {nom_model} - R2 = {r2_test}")
    # plt.show()

    file_out = f"{path_output}/residus_ytrain_{nom_model}.png"
    fig.savefig(file_out)

    # Center réduire les résidus et afficher le QQ-plot

    fig = plt.figure()
    residus_norm = (residus - residus.mean()) / residus.std()
    probplot(residus_norm, plot=plt)
    plt.title(f"{var_cible} - {nom_model} - R2 = {r2_test}")
    # plt.show()

    file_out = f"{path_output}/QQ_plot_residus_ytrain_{nom_model}.png"
    fig.savefig(file_out)

    # Les valeurs ajustées et les résidus du modèle -> afficher les résidus
    pred_test = lr.predict(X_test)
    residus = pred_test - y_test

    fig = plt.figure()
    plt.scatter(y_test, residus)
    plt.plot((y_test.min(), y_test.max()), (0, 0), lw=3, color='#0a5798')
    plt.xlabel("y_test")
    plt.ylabel("residus")
    plt.title(f"{var_cible} - {nom_model} - R2 test = {r2_test}")
    # plt.show()

    file_out = f"{path_output}/residus_ytest_{nom_model}.png"
    fig.savefig(file_out)

    # Center réduire les résidus et afficher le QQ-plot

    fig = plt.figure()
    residus_norm = (residus - residus.mean()) / residus.std()
    probplot(residus_norm, plot=plt)
    plt.title(f"{var_cible} - {nom_model} - R2 test = {r2_test}")
    # plt.show()

    file_out = f"{path_output}/QQ_plot_residus_ytest_{nom_model}.png"
    fig.savefig(file_out)

X_train, X_test, y_train, y_test = load(data_file)

linear_regression(X_train, X_test, y_train, y_test, nom_model="All_var")

# ========================================================================================
# Modèle Affiné
# ========================================================================================
# enlever la variable Temp9am
var_enlever = "Temp9am"
X_train_affine = X_train.drop(var_enlever, axis=1)
X_test_affine = X_test.drop(var_enlever, axis=1)
linear_regression(X_train_affine, X_test_affine, y_train, y_test, nom_model=f"Sans_{var_enlever}")

# ========================================================================================
# SELECT KBEST
# ========================================================================================

from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns

data = pd.concat([X_train, X_test], axis=0)
target = pd.concat([y_train, y_test])

df = data.copy()
df[var_cible] = target
corr_mat = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_mat, annot=True, center=0, cmap='RdBu_r')
plt.show()

sk = SelectKBest(f_regression, k=8)
sk.fit(data, target)

selected = sk.get_support()

df_var = pd.DataFrame({'variable': data.columns,
                       'selected': selected})
print(df_var)

sk_train = sk.transform(X_train)
sk_test = sk.transform(X_test)

lr3 = LinearRegression()
lr3.fit(sk_train, y_train)

r2_train = lr3.score(sk_train, y_train)
r2_test = lr3.score(sk_test, y_test)

print(r2_train)
print(r2_test)

from sklearn.feature_selection import SelectFromModel

lr = LinearRegression()

sfm = SelectFromModel(lr)

sfm_train = sfm.fit_transform(X_train, y_train)
sfm_test = sfm.transform(X_test)

sfmlr = LinearRegression()
sfmlr.fit(sfm_train, y_train)

print(sfmlr.score(sfm_train, y_train))
print(sfmlr.score(sfm_test, y_test))

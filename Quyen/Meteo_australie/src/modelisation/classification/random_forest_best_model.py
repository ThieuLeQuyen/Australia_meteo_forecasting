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
import xgboost as xgb
from hyperopt import hp
from joblib import dump
from joblib import load
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from context.context import ctx
from src.modelisation.classification.classification_evaluation import ClassificationEvaluation
from src.modelisation.classification.optimization_hyperparameters import HyperOpt

os.chdir("/Users/quyen/PycharmProjects/Meteo_australie")
print(os.getcwd())
model = "random_forest"
metric_name = "accuracy"

k = 4
with_location = False

log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}_early_stopping"

ctx.cmd_start(file_log_name=log_file, verbose_level="min")

if with_location:
    path_classification = f"./output/classification/knn_impute_{k}_with_location"
    path_output = os.path.abspath(os.sep.join([path_classification, model]))
    data_file = f"./output/modelisation/data_original_pre_processing_with_location_knn_imputed_{k}.joblib"
else:
    path_classification = f"./output/classification/knn_impute_{k}"
    path_output = os.path.abspath(os.sep.join([path_classification, model]))
    data_file = f"./output/modelisation/data_original_pre_processing_knn_imputed_{k}.joblib"

os.makedirs(path_output, exist_ok=True)

X_train, X_test, y_train, y_test = load(data_file)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

# ================================================================================================
# Hyperopt pour optimiser les hyperparameters
# ================================================================================================

ctx.log_min("Hyperopt pour optimiser les hyperparameters")

params_file = f"{path_classification}/{model}/best_params_{metric_name}.joblib"
best_params = load(params_file)

clf_rf = ensemble.RandomForestClassifier(**best_params)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

accuracy = cross_val_score(clf_rf, X=X_train, y=y_train_encoded, cv=cv, scoring="accuracy").mean()
f1 = cross_val_score(clf_rf, X=X_train, y=y_train_encoded, cv=cv, scoring="f1").mean()

# =============================================================================================
# Modèle avec que des features importants
# =============================================================================================

feats = {}
for feature, importance in zip(X_train.columns, clf_rf.feature_importances_):
    feats[feature] = importance

df_importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
df_importance = df_importance.sort_values(by="Importance", ascending=False)

list_feature_to_hold = df_importance[df_importance["Importance"] >= 0.004].index.to_list()

X_train_reduce = X_train[list_feature_to_hold]
X_test_reduce = X_test[list_feature_to_hold]

clf_rf_reduce = ensemble.RandomForestClassifier(**best_params)

clf_rf_reduce.fit(X_train_reduce, y_train_encoded)
clf_rf_reduce.score(X_test_reduce, y_test_encoded)

# =============================================================================================
# Validation curve pour tree decision avec yellowbrick
# =============================================================================================

# min_depth = 1
# max_depth = 15
# list_depth = np.arange(min_depth, max_depth, 1)
#
# criterion = 'gini'
#
# dt_clf = tree.DecisionTreeClassifier(criterion=criterion)
# cv = StratifiedKFold(12)
#
# title = f"Validation Curve pour DecisionTreeClassifier - criterion = {criterion}"
# file_out = f"{path_output}/Validation_curve_{model}_criterion_{criterion}.png"
#
# fig, ax = plt.subplots()
# viz = validation_curve(estimator=dt_clf,
#                        X=pd.concat([X_train, X_test], axis=0),
#                        y=pd.concat([y_train, y_test], axis=0),
#                        param_name='max_depth',
#                        param_range=list_depth,
#                        scoring='accuracy', cv=cv, ax=ax, n_jobs=6,
#                        title=title)
# viz.show(outpath=file_out)

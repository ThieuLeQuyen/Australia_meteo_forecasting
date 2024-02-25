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
from hyperopt import hp
from joblib import load
from matplotlib import pyplot as plt
from sklearn import ensemble, tree
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from yellowbrick.model_selection import validation_curve
from joblib import dump

from src.modelisation.classification.classification_evaluation import ClassificationEvaluation
from src.modelisation.classification.optimization_hyperparameters import HyperOpt
from context.context import ctx

os.chdir("/Users/quyen/PycharmProjects/Meteo_australie")
print(os.getcwd())
model = "voting_model"

with_early_stopping = True

k = 4
with_location = False

log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}.log"

ctx.cmd_start(file_log_name=log_file, verbose_level="min")

if with_location:
    path_classification = f"./output/classification/knn_impute_{k}_with_location"
    path_output = os.path.abspath(os.sep.join([path_classification, model]))
    data_file = f"./output/modelisation/data_pre_processing_with_location_knn_imputed_{k}.joblib"
else:
    path_classification = f"./output/classification/knn_impute_{k}"
    path_output = os.path.abspath(os.sep.join([path_classification, model]))
    data_file = f"./output/modelisation/data_pre_processing_knn_imputed_{k}.joblib"

os.makedirs(path_output, exist_ok=True)

X_train, X_test, y_train, y_test = load(data_file)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

# ================================================================================================
# Récupérer les best models de LogisticRegression, KNN, TreeDecision, RandomForest et XGBoost
# ================================================================================================
metric_name = "accuracy_score"

ctx.log_min("Récupérer les best models de LogisticRegression, KNN, TreeDecision, RandomForest et XGBoost")

# LogisticRegression
data_file = f"{path_classification}/logistic_regression/best_params_{metric_name}.joblib"
best_params = load(data_file)
clf_lr = LogisticRegression(**best_params, max_iter=10, solver='saga')

# TreeDecision
data_file = f"{path_classification}/tree_decision/best_params_{metric_name}.joblib"
best_params = load(data_file)
clf_tree = tree.DecisionTreeClassifier(**best_params)

# RandomForest
data_file = f"{path_classification}/random_forest/best_params_{metric_name}.joblib"
best_params = load(data_file)
clf_rf = ensemble.RandomForestClassifier(**best_params)

# XGBoost
data_file = f"{path_classification}/xg_boost/best_params_{metric_name}.joblib"
best_params = load(data_file)
clf_xgboost = xgb.XGBClassifier(**best_params, random_state=42)

# =================================================================================================
# Voting classifier
# =================================================================================================

vclf = VotingClassifier(estimators=[('lr', clf_lr), ('tree', clf_tree), ('rf', clf_rf), ('xgb', clf_xgboost)],
                        voting='hard')

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for clf, label in zip([clf_lr, clf_tree, clf_rf, clf_xgboost, vclf],
                      ['Logistic Regression', 'Tree Decision', 'Random Forest', 'XGBoost', 'Voting']):
    scores = cross_validate(clf, X_train, y_train_encoded, cv=cv, scoring=["accuracy", "f1"])
    print("[%s]: \n Accuracy: %0.2f (+/- %0.2f)" % (label, scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
          "F1 score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))
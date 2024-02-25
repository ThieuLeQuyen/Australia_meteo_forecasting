# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""
import os
import time

import numpy as np
from hyperopt import hp
from joblib import dump
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from context.context import ctx
from src.modelisation.classification.classification_evaluation import ClassificationEvaluation
from src.modelisation.classification.optimization_hyperparameters import HyperOpt

print(os.getcwd())

k = 4
model = "logistic_regression"
with_location = True

if with_location:
    path_output = f"./output/classification/knn_impute_{k}_with_location/{model}"
    data_file = f"./output/classification/data_scaled_pre_processing_with_location_knn_imputed_{k}.joblib"
    log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}_with_location.log"
else:
    path_output = f"./output/classification/knn_impute_{k}/{model}"
    data_file = f"./output/classification/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib"
    log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}.log"

os.makedirs(path_output, exist_ok=True)

ctx.cmd_start(file_log_name=log_file, verbose_level="min")

X_train, X_test, y_train, y_test = load(data_file)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

# ================================================================================================
# Hyperopt pour optimiser les hyperparameters
# ================================================================================================

ctx.log_min("Hyperopt pour optimiser les hyperparameters")

param_space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
}

list_metric_considered = ['accuracy', 'recall', 'precision', 'f1', "roc_auc"]

for metric_considered in list_metric_considered:

    # metric_name = metric_considered.__name__
    ctx.log_min(f"=========== {metric_considered} ===========")

    t1 = time.time()

    # solver='saga' pour permettre à utiliser 'l1' pour 'penalty'
    clf = LogisticRegression(max_iter=200, solver='saga')
    ho = HyperOpt(clf)
    result_hyperopt = ho.optim(param_space,
                               X_train=X_train, y_train=y_train_encoded,
                               X_test=X_test, y_test=y_test_encoded,
                               max_evals=200, metric=metric_considered,
                               with_cross_validation=True)
    t2 = time.time()

    ctx.log_min("temps de HyperOpt = ", t2 - t1)

    best = result_hyperopt['best']
    best_params = {
        'C': best['C'],
        'penalty': ['l1', 'l2'][best['penalty']],
        'fit_intercept': [True, False][best['fit_intercept']],
    }

    df_trials = result_hyperopt['trials']

    # Sauvegarder les hyperparameters optimaux et les résultats des trials
    ctx.log_min("Sauvegarder les hyperparameters optimaux et les résultats des trials")

    file_out = f"{path_output}/best_params_{metric_considered}.joblib"
    dump(best_params, file_out)

    file_out = f"{path_output}/trials_{metric_considered}.csv"
    df_trials.to_csv(file_out, sep=";")

    # ================================================================================================
    # Evaluation du meilleur modèle par hyperopt
    # ================================================================================================
    ctx.log_min("Evaluation du meilleur modèle par hyperopt")

    clf = LogisticRegression(**best_params, max_iter=10, solver='saga')

    ce = ClassificationEvaluation(clf, X_train, y_train_encoded, X_test, y_test_encoded)

    score = round(ce.score(), 4)

    model_name = f"{model}_hyperopt_{metric_considered}"

    file_confusion_matrix = f"{path_output}/confusion_matrix_{model_name}.png"
    file_class_report = f"{path_output}/class_report_{model_name}.png"
    file_rocauc = f"{path_output}/ROCAUC_{model_name}.png"
    file_feat_import = f"{path_output}/Feature_importances_{model_name}.png"

    title_confusion_matrix = f"LogisticRegression Confusion Matrix - accuracy = {score}"
    title_class_report = f"LogisticRegression Classification Report - accuracy = {score}"
    title_rocauc = f"ROC Curves for LogisticRegression - accuracy = {score}"
    title_feat_import = f"Feature Importances using LogisticRegression - accuracy = {score}"

    ce.evaluate(classes=classes,
                file_confusion_matrix=file_confusion_matrix,
                file_class_report=file_class_report,
                file_rocauc=file_rocauc,
                file_feat_import=file_feat_import,
                title_confusion_matrix=title_confusion_matrix,
                title_class_report=title_class_report,
                title_rocauc=title_rocauc,
                title_feat_import=title_feat_import)


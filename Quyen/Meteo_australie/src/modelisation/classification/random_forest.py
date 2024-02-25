# -*- coding: utf-8 -*-

"""
@created: 10/2023
@updated:
@author: quyen@marcaud.fr
"""
import os
import time

import numpy as np
from hyperopt import hp
from joblib import dump
from joblib import load
from sklearn import ensemble
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from context.context import ctx
from src.modelisation.classification.classification_evaluation import ClassificationEvaluation
from src.modelisation.classification.optimization_hyperparameters import HyperOpt

print(os.getcwd())
model = "random_forest"
k = 6

with_location = False

if with_location:
    path_output = f"./output/classification/knn_impute_{k}_with_location/{model}"
    data_file = f"./output/classification/data_original_pre_processing_with_location_knn_imputed_{k}.joblib"
    log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}_with_location.log"
else:
    path_output = f"./output/classification/knn_impute_{k}/{model}"
    data_file = f"./output/classification/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"
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

max_depth_range = np.arange(2, 14+1, dtype=int)
min_samples_split_range = np.arange(2, 9+1, dtype=int)
min_samples_leaf_range = np.arange(2, 9+1, dtype=int)
n_estimators_range = np.arange(10, 20, dtype=int)

ctx.log_min("max_depth_range: ", max_depth_range)
ctx.log_min("min_samples_split_range: ", min_samples_split_range)
ctx.log_min("min_samples_leaf_range: ", min_samples_leaf_range)
ctx.log_min("n_estimators_range: ", n_estimators_range)

param_space = {
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),
    'max_depth': hp.choice('max_depth', max_depth_range),
    'min_samples_split': hp.choice('min_samples_split', min_samples_split_range),
    'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf_range),
    'n_estimators': hp.choice('n_estimators', n_estimators_range),
    'bootstrap': hp.choice('bootstrap', [True, False])
}

list_metric_considered = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']

for metric_considered in list_metric_considered:

    ctx.log_min(f"=========== {metric_considered} ===========")

    t1 = time.time()
    clf = ensemble.RandomForestClassifier()
    ho = HyperOpt(clf)
    result_hyperopt = ho.optim(param_space, X_train, y_train_encoded, X_test, y_test_encoded,
                               max_evals=250, metric=metric_considered)
    t2 = time.time()

    ctx.log_min("temps de HyperOpt = ", t2 - t1)

    best = result_hyperopt['best']
    best_params = {
        'criterion': ['gini', 'entropy'][best['criterion']],
        'max_features': [None, 'sqrt', 'log2'][best['max_features']],
        'max_depth': max_depth_range[best['max_depth']],
        'min_samples_split': min_samples_split_range[best['min_samples_split']],
        'min_samples_leaf': min_samples_leaf_range[best['min_samples_leaf']],
        'n_estimators': n_estimators_range[best['n_estimators']],
        'bootstrap': [True, False][best['bootstrap']]
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

    clf = ensemble.RandomForestClassifier(**best_params)

    ce = ClassificationEvaluation(clf, X_train, y_train_encoded, X_test, y_test_encoded)

    score = round(ce.score(), 4)

    model_name = f"{model}_hyperopt_{metric_considered}"
    file_confusion_matrix = f"{path_output}/confusion_matrix_{model_name}.png"
    file_class_report = f"{path_output}/class_report_{model_name}.png"
    file_rocauc = f"{path_output}/ROCAUC_{model_name}.png"
    file_feat_import = f"{path_output}/Feature_importances_{model_name}.png"

    title_confusion_matrix = f"RandomForest Confusion Matrix - accuracy = {score}"
    title_class_report = f"RandomForest Classification Report - accuracy = {score}"
    title_rocauc = f"ROC Curves for RandomForest - accuracy = {score}"
    title_feat_import = f"Feature Importances using RandomForest - accuracy = {score}"

    ce.evaluate(classes=classes,
                file_confusion_matrix=file_confusion_matrix,
                file_class_report=file_class_report,
                file_rocauc=file_rocauc,
                file_feat_import=file_feat_import,
                title_confusion_matrix=title_confusion_matrix,
                title_class_report=title_class_report,
                title_rocauc=title_rocauc,
                title_feat_import=title_feat_import
                )

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

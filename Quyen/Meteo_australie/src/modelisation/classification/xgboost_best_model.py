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
from joblib import load
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import shap

from context.context import ctx
from src.modelisation.classification.classification_evaluation import ClassificationEvaluation

os.chdir("/Users/quyen/PycharmProjects/Meteo_australie")
print(os.getcwd())
model = "xg_boost"

metric_name = "accuracy"

k = 4
with_location = False

log_file = time.strftime("%Y%m%d-%H%M%S-", time.localtime(time.time())) + f"{model}_early_stopping"

ctx.cmd_start(file_log_name=log_file, verbose_level="min")

if with_location:
    path_classification = f"./output/classification/knn_impute_{k}_with_location"
    path_output = os.path.abspath(os.sep.join([path_classification, model]))
    data_file = f"./output/classification/data_original_pre_processing_with_location_knn_imputed_{k}.joblib"
else:
    path_classification = f"./output/classification/knn_impute_{k}"
    path_output = os.path.abspath(os.sep.join([path_classification, model]))
    data_file = f"./output/classification/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"

os.makedirs(path_output, exist_ok=True)

X_train, X_test, y_train, y_test = load(data_file)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

# ================================================================================================
# Default model
# ================================================================================================

xgb_default = xgb.XGBClassifier()

xgb_default.fit(X_train, y_train_encoded)
y_pred_df = xgb_default.predict(X_test)
xgb_default.score(X_test, y_test_encoded)

auc_default = round(metrics.roc_auc_score(y_test_encoded, xgb_default.predict_proba(X_test)[:, 1]), 4)

# ================================================================================================
# Best model par hyperopt
# ================================================================================================

ctx.log_min("Hyperopt pour optimiser les hyperparameters")

params_file = f"{path_classification}/xg_boost/best_params_{metric_name}.joblib"
best_params = load(params_file)

xgb_best = xgb.XGBClassifier(**best_params, early_stopping_rounds=50, random_state=42)

xgb_best.fit(X_train, y_train_encoded,
             eval_set=[(X_train, y_train_encoded), (X_test, y_test_encoded)])
print(xgb_best.score(X_test, y_test_encoded))
print(xgb_best.score(X_train, y_train_encoded))

auc_best = round(metrics.roc_auc_score(y_test_encoded, xgb_best.predict_proba(X_test)[:, 1]), 4)

xgb_best.best_ntree_limit

# =============================================================================================
# Validation curve pour tree decision avec yellowbrick
# =============================================================================================

fig, ax = plt.subplots()
metrics.RocCurveDisplay.from_estimator(xgb_default, X_test, y_test_encoded,
                                       ax=ax, label=f'default model - AUC = {auc_default}')
metrics.RocCurveDisplay.from_estimator(xgb_best, X_test, y_test_encoded,
                                       ax=ax, label=f'best model - AUC = {auc_best}')
ax.set_title("ROC Curve des modèles XGBoost")
plt.show()

# =============================================================================================
# SHAP - waterfall
# =============================================================================================
shap_explainer = shap.TreeExplainer(xgb_best)
shap_values = shap_explainer(X_test)
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

res = pd.concat([shap_df.sum(axis='columns').rename('pred') + shap_values.base_values,
                 pd.Series(y_test_encoded, name='true')],
                axis='columns').assign(prob=lambda adf: (np.exp(adf.pred) / (1 + np.exp(adf.pred))))

s = 0
for s in range(5):
    plt.figure()
    shap.plots.waterfall(shap_values[s], max_display=20, show=False)
    plt.tight_layout()

    file_out = f'{path_output}/waterfall_sample_{s}.png'
    plt.savefig(file_out)

# =============================================================================================
# SHAP - Mean SHAP Plot
# =============================================================================================

plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
file_out = f'{path_output}/mean_SHAP_barplot.png'
plt.savefig(file_out)

# =============================================================================================
# SHAP - Beeswarm Plot
# =============================================================================================

plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
file_out = f'{path_output}/beeswarm.png'
plt.savefig(file_out)

# =============================================================================================
# SHAP - Dependence Plots
# =============================================================================================

list_variables = ["Humidity3pm", "Pressure3pm", "WindGustSpeed", "Sunshine", "RainToday", "Rainfall",
                  "MaxTemp"]
for variable in list_variables:
    plt.figure()
    shap.plots.scatter(shap_values[:, variable], show=False)
    plt.tight_layout()
    file_out = f'{path_output}/dependence_{variable}.png'
    plt.savefig(file_out)

###################################################
list_metric_considered = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
clf = xgb_best

for metric_considered in list_metric_considered:
    model_name = f"{model}_hyperopt_{metric_considered}_early_stopping"

    ce = ClassificationEvaluation(clf, X_train, y_train_encoded, X_test, y_test_encoded)

    score = round(ce.score(early_stopping_rounds=50), 4)

    file_confusion_matrix = f"{path_output}/confusion_matrix_{model_name}.png"
    file_class_report = f"{path_output}/class_report_{model_name}.png"
    file_rocauc = f"{path_output}/ROCAUC_{model_name}.png"
    file_feat_import = f"{path_output}/Feature_importances_{model_name}.png"

    title_confusion_matrix = f"XGBoost Confusion Matrix - accuracy = {score}"
    title_class_report = f"XGBoost Classification Report - accuracy = {score}"
    title_rocauc = f"ROC Curves for XGBoost - accuracy = {score}"
    title_feat_import = f"Feature Importances using XGBoost - accuracy = {score}"

    ce.evaluate(classes=classes, early_stopping_rounds=50,
                file_confusion_matrix=file_confusion_matrix,
                file_class_report=file_class_report,
                file_rocauc=file_rocauc,
                file_feat_import=file_feat_import,
                title_confusion_matrix=title_confusion_matrix,
                title_class_report=title_class_report,
                title_rocauc=title_rocauc,
                title_feat_import=title_feat_import,
                confusion_matrix=False,
                class_report=False,
                rocauc=False,
                feature_importance=True)

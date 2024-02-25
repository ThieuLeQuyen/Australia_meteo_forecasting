# -*- coding: utf-8 -*-

"""
@created: 10/2023
@updated:
@author: quyen@marcaud.fr
"""
import os

import pandas as pd
import xgboost as xgb
from joblib import load
from matplotlib import pyplot as plt
from sklearn import ensemble, tree, metrics
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

os.chdir("/Users/quyen/PycharmProjects/Meteo_australie")
print(os.getcwd())

with_early_stopping = True

k = 4
with_location = False

if with_location:
    path_classification = f"./output/classification/knn_impute_{k}_with_location"
    data_original = f"./output/classification/data_original_pre_processing_with_location_knn_imputed_{k}.joblib"
    data_scaled = f"./output/classification/data_scaled_pre_processing_with_location_knn_imputed_{k}.joblib"
else:
    path_classification = f"./output/classification/knn_impute_{k}"
    data_original = f"./output/classification/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"
    data_scaled = f"./output/classification/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib"

path_output = path_classification

# X_train_ori, X_test_ori, y_train, y_test = load(data_original)
X_train, X_test, y_train, y_test = load(data_scaled)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

# ================================================================================================
# Récupérer les best models de LogisticRegression, KNN, TreeDecision, RandomForest et XGBoost
# ================================================================================================

list_metric_considered = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
list_model = ["LogisticRegression", "TreeDecision", "RandomForest", "XGBoost"]

for metric_name in list_metric_considered:
    print(f"=============== {metric_name} ==============")
    # LogisticRegression
    data_file = f"{path_classification}/logistic_regression/best_params_{metric_name}.joblib"
    best_params = load(data_file)
    clf_lr = LogisticRegression(**best_params, max_iter=10, solver='saga')

    # # KNN
    # data_file = f"{path_classification}/knn/best_params_{metric_name}.joblib"
    # best_params = load(data_file)
    # clf_knn = neighbors.KNeighborsClassifier(**best_params)

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
    # Calculer les scores
    # =================================================================================================
    df_scores = pd.DataFrame(columns=list_metric_considered, index=list_model)
    fig_rain, ax_rain = plt.subplots()
    fig_no_rain, ax_no_rain = plt.subplots()

    for clf, label in zip([clf_lr, clf_tree, clf_rf, clf_xgboost], list_model):
        print(label)
        clf.fit(X_train, y_train_encoded)

        y_pred = clf.predict(X_test.values)

        accuracy = accuracy_score(y_test_encoded, y_pred)
        recall = recall_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred)
        rocauc_rain = round(roc_auc_score(y_test_encoded, clf.predict_proba(X_test)[:, 1]), 4)
        rocauc_no_rain = round(roc_auc_score(y_test_encoded, clf.predict_proba(X_test)[:, 0]), 4)

        df_scores.loc[label, :] = [accuracy, recall, precision, f1, rocauc_rain]

        metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test_encoded,
                                               ax=ax_rain, label=f'{label} - AUC Rain = {rocauc_rain}')
        metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test_encoded, pos_label=0,
                                               ax=ax_no_rain, label=f'{label} - AUC Rain = {rocauc_no_rain}')

    df_scores["Model"] = df_scores.index
    ax_rain.set_title("Comparaison des ROC Curve pour Rain")
    ax_no_rain.set_title("Comparaison des ROC Curve pour No Rain")

    if with_location:
        file_scores = f"{path_output}/Scores_of_modeles_with_location_best_{metric_name}.csv"
        file_rocauc_rain = f"{path_output}/ROCCurve_of_Rain_of_modeles_with_location_best_{metric_name}.png"
        file_rocauc_no_rain = f"{path_output}/ROCCurve_of_No_Rain_of_modeles_with_location_best_{metric_name}.png"
    else:
        file_scores = f"{path_output}/Scores_of_modeles_best_{metric_name}.csv"
        file_rocauc_rain = f"{path_output}/ROCCurve_of_Rain_of_modeles_best_{metric_name}.png"
        file_rocauc_no_rain = f"{path_output}/ROCCurve_of_No_Rain_of_modeles_best_{metric_name}.png"

    df_scores.to_csv(file_scores, sep=";")
    fig_rain.savefig(file_rocauc_rain)
    fig_no_rain.savefig(file_rocauc_no_rain)

# -*- coding: utf-8 -*-

import os

import dtreeviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from yellowbrick import classifier
from yellowbrick.model_selection import validation_curve
from sklearn import ensemble
from joblib import load

from src.modelisation.classification.classification_evaluation import ClassificationEvaluation

print(os.getcwd())

k = 4
X_train, X_test, y_train, y_test = load(f"./output/modelisation/data_pre_processing_knn_imputed_{k}.joblib")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

# =============================================================================================
# Random Forest
# =============================================================================================
# start = time.time()
# rf = ensemble.RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train_encoded)
# rf.score(X_test, y_test_encoded)
# end = time.time()
#
# print("temps = ", end - start)

#
# fig, ax = plt.subplots()
#
# tree.plot_tree(rf.estimators_[0],
#                feature_names=list(X_train.columns),
#                filled=True,
#                class_names=list(rf.classes_),
#                ax=ax,
#                max_depth=3)
#
# plt.show()

# Classification report
rf = ensemble.RandomForestClassifier(random_state=42)
ce = ClassificationEvaluation(rf, X_train, y_train_encoded, X_test, y_test_encoded)
ce.viz_classification_report(file_out="./output/classification/test.png", support=True)
# ce.viz_classification_report()

# Matrice de confusion

ce.viz_confusion_matrix(file_out="./output/classification/cm.png",
                        normalize=True, classes=classes)
ce.viz_confusion_matrix(normalize=True, classes=classes)

cm = classifier.ConfusionMatrix(rf, classes=classes, percent=True,
                                title="AAAA", is_fitted='auto')
cm.fit(X_train, y_train_encoded)
cm.score(X_test, y_test_encoded)
cm.show()

# ROC curve

roc_curve = classifier.ROCAUC(rf, classes=classes)
roc_curve.fit(X_train, y_train_encoded)
roc_curve.score(X_test, y_test_encoded)
roc_curve.show()

# =============================================================================================
# XGBoost Random Forest
# =============================================================================================

rf_xgb = xgb.XGBRFClassifier(random_state=42)
rf_xgb.fit(X_train, y_train_encoded)
rf_xgb.score(X_test, y_test_encoded)

fig, ax = plt.subplots()
xgb.plot_tree(rf_xgb, num_trees=0, ax=ax)
plt.show()

# Use dtreeviz
fig, ax = plt.subplots()
viz_dtree = dtreeviz.model(rf_xgb, X_train=X_train, y_train=y_train_encoded,
                           target_name='Rain tomorrow',
                           feature_names=list(X_train.columns),
                           class_names=["No Rain", "Rain"], tree_index=0
                           )
viz_dtree.view(depth_range_to_display=(0, 2))
plt.show()

# =============================================================================================
# Training the number of trees in the forest
# =============================================================================================

list_number_of_tree = np.arange(1, 100, 2)
# list_number_of_tree = [5, 10, 15, 20, 25, 30, 35, 40 ,45, 50 ,55 ,60]
fig, ax = plt.subplots()
viz = validation_curve(xgb.XGBRFClassifier(random_state=42),
                       X=pd.concat([X_train, X_test], axis=0),
                       y=np.concatenate([y_train_encoded, y_test_encoded]),
                       param_name='n_estimators',
                       param_range=list_number_of_tree,
                       scoring='accuracy', cv=3, ax=ax
                       )
plt.show()

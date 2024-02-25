# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load
from sklearn import tree, dummy, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yellowbrick.model_selection import validation_curve
from yellowbrick import classifier
from sklearn.model_selection import GridSearchCV
import dtreeviz

from context.context import Context

k = 4
X_train, X_test, y_train, y_test = load(f"./output/modelisation/data_pre_processing_knn_imputed_{k}.joblib")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# =============================================================================================
# Stumps tree
# =============================================================================================

dt_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=123)

dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)

# Visualisation de tree

fig, ax = plt.subplots()
features = list(c for c in X_train.columns)
tree.plot_tree(dt_clf, feature_names=features,
               filled=True,
               class_names=list(dt_clf.classes_),
               ax=ax)
plt.show()

y_pred = dt_clf.predict(X_test)

result_eval = metrics.classification_report(y_test, y_pred)
print(result_eval)

cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')

fig, ax = plt.subplots()
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap="Blues")
plt.show()

fig, ax = plt.subplots()
dt_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=123)
vizualizer = classifier.classification_report(dt_clf, X_train, y_train, classes=["Rain", "No Rain"], support=True)

vizualizer.fit(X_train, y_train)
vizualizer.score(X_test, y_test)
vizualizer.show()

# =============================================================================================
# Validation curve pour tree decision
# =============================================================================================

accuracies = []

min_depth = 1
max_depth = 15
list_depth = np.arange(min_depth, max_depth, 1)

for depth in list_depth:
    clf_tree = tree.DecisionTreeClassifier(max_depth=depth)
    clf_tree.fit(X_train, y_train)
    accuracies.append(clf_tree.score(X_test, y_test))

fig = plt.figure()
plt.plot(list_depth, accuracies, "-*")
plt.xticks(list_depth)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title("Accuracy at a given Tree Depth")
plt.show()

# =============================================================================================
# Validation curve pour tree decision avec yellowbrick
# =============================================================================================

fig, ax = plt.subplots()

dt_clf = tree.DecisionTreeClassifier(criterion='gini')
viz = validation_curve(estimator=dt_clf,
                       X=pd.concat([X_train, X_test], axis=0),
                       y=pd.concat([y_train, y_test], axis=0),
                       param_name='max_depth',
                       param_range=list_depth,
                       scoring='accuracy', cv=5, ax=ax, n_jobs=6
                       )
viz.set_title("Validation Curve pour DecisionTreeClassifier - criterion = gini")
plt.show()

fig, ax = plt.subplots()

dt_clf = tree.DecisionTreeClassifier(criterion='entropy')
viz = validation_curve(estimator=dt_clf,
                       X=pd.concat([X_train, X_test], axis=0),
                       y=pd.concat([y_train, y_test], axis=0),
                       param_name='max_depth',
                       param_range=list_depth,
                       scoring='accuracy', cv=5, ax=ax, n_jobs=6
                       )
viz.set_title("Validation Curve pour DecisionTreeClassifier - criterion = entropy")
plt.show()

# =============================================================================================
# Grid Search Hyper Parameters
# =============================================================================================

params = {
    'max_depth': [5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 3, 4, 5, 6],
}

grid_search = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                           param_grid=params,
                           cv=4,
                           n_jobs=-1,
                           verbose=1, scoring='accuracy')

grid_search.fit(X=pd.concat([X_train, X_test]), y=pd.concat([y_train, y_test]))
best_params = grid_search.best_params_

best_tree = tree.DecisionTreeClassifier(**best_params)
best_tree.fit(X_train, y_train)
best_tree.score(X_test, y_test)

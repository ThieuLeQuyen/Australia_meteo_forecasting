# -*- coding: utf-8 -*-

import os

import dtreeviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load
from sklearn import tree, dummy, preprocessing, metrics
from sklearn.model_selection import train_test_split
from yellowbrick.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble

from src.modelisation.utils import dot_export

k = 4
X_train, X_test, y_train, y_test = load(f"./output/modelisation/data_pre_processing_knn_imputed_{k}.joblib")

label_encoder = preprocessing.LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# =============================================================================================
# Creating an XGBoost Model
# =============================================================================================

# Combined Data for cross validation
X = pd.concat([X_train, X_test], axis='index')
y = pd.Series([*y_train_encoded, *y_test_encoded], index=X.index)

xg_oob = xgb.XGBClassifier()
xg_oob.fit(X_train, y_train_encoded)
xg_oob.score(X_test, y_test_encoded)

y_pred = xg_oob.predict(X_test)

result_eval = metrics.classification_report(y_test_encoded, y_pred)
print(result_eval)

# =============================================================================================
#
# =============================================================================================

xg2 = xgb.XGBClassifier(max_depth=2, n_estimators=2)
xg2.fit(X_train, y_train_encoded)
xg2.score(X_test, y_test_encoded)

# Let's look at what the first tree looks like

index_tree = 0

fig = plt.figure()
viz = dtreeviz.model(xg2, X_train=X, y_train=y,
                     target_name="Rain Tomorrow",
                     feature_names=list(X.columns),
                     class_names=["No rain", 'Rain'], tree_index=index_tree)
viz.view(depth_range_to_display=(0, 2))

plt.show()

#

dot_export(xg2, num_trees=0, filename="./output/xgb_oob.dot", title="First tree")

# =============================================================================================
# Early stopping
# =============================================================================================

xg_es = xgb.XGBClassifier(early_stopping=20)
xg_es.fit(X_train, y_train_encoded,
          eval_set=[(X_train, y_train_encoded), (X_test, y_test_encoded)])
xg_es.score(X_test, y_test_encoded)

xg_es.best_ntree_limit

results = xg_es.evals_result()

fig, ax = plt.subplots()
ax = (pd.DataFrame({'training': results['validation_0']['logloss'],
                    'testing': results['validation_1']['logloss']
                    })
      .assign(ntrees=lambda adf: range(1, len(adf) + 1))
      .set_index('ntrees')
      .plot(figsize=(5, 4), ax=ax, title='eval_results with early stopping')
      )
ax.set_xlabel('ntrees')
plt.show()


xg_err = xgb.XGBClassifier(early_stopping=20, eval_metric='error')
xg_err.fit(X_train, y_train_encoded,
          eval_set=[(X_train, y_train_encoded), (X_test, y_test_encoded)])
xg_err.score(X_test, y_test_encoded)

xg_err.best_ntree_limit

# =============================================================================================
# Tuning Hyperparameters
# =============================================================================================

fig, ax = plt.subplots()

validation_curve(estimator=xgb.XGBClassifier(),
                 X=X_train, y=y_train_encoded,
                 param_name='gamma',
                 param_range=[0, 0.5, 1.5, 5, 10, 20, 30],
                 ax=ax)

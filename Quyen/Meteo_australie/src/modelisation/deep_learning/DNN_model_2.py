# -*- coding: utf-8 -*-

"""
@created: 11/2023
@updated:
@author: quyen@marcaud.fr
"""
import os
import time

import numpy as np
import tensorflow as tf
from joblib import load
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from scikeras.wrappers import KerasClassifier


from context.context import ctx

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
# Définir les métriques
# ================================================================================================

def recall(y_true, y_pred):
    y_true = tf.ones_like(y_true)
    true_positives = tf.sum(tf.round(tf.clip(y_true * y_pred, 0, 1)))
    all_positives = tf.sum(tf.round(tf.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + tf.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = tf.ones_like(y_true)
    true_positives = tf.sum(tf.round(tf.clip(y_true * y_pred, 0, 1)))

    predicted_positives = tf.sum(tf.round(tf.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.epsilon()))


# ================================================================================================
# Default model
# ================================================================================================
n_var_expl = X_train.shape[1]

# Instancier une couche Input, avec pour dimension le nombre de variables explicatives du modèles.
inputs = Input(shape=n_var_expl, name="Input")

# Instancier une couche Dense, avec 1 seul neurone.
dense = Dense(units=1, activation="softmax")

# Appliquez la couche Dense à la couche d'Input.
outputs = dense(inputs)

# Définissez dans une variable linear_model le modèle à l'aide de la fonction Model en utilisant pour les argument inputs et outputs les résultats obtenus précédemment.
model = Model(inputs=inputs, outputs=outputs)

# Affichez le résumé du modèle
model.summary()

# Compiler le modèle
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=[recall])

model.fit(X_train, y_train_encoded, epochs=500, batch_size=32, validation_split=0.1)

y_pred = model.predict(X_test)

y_test_class = y_test_encoded
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

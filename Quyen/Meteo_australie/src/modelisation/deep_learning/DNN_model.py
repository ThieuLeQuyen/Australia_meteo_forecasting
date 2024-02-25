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
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasClassifier
from scipy.stats import reciprocal
import matplotlib.pyplot as plt

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# =================================================================================================
#
# =================================================================================================
def plot_train_validation(epochs, train_loss_values, validation_loss_values,
                          train_acc_values, validation_acc_values):
    fig, ax = plt.subplots(figsize=(10, 10), ncols=1, nrows=2)
    ax[0].plot(epochs, train_loss_values, color="green", label='Training loss')
    ax[0].plot(epochs, validation_loss_values, color="red", label='Validation loss')
    ax[0].set_title('Training and validation loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, train_acc_values, color="blue", label='Training acc')
    ax[1].plot(epochs, validation_acc_values, color="violet", label='Validation acc')
    ax[1].set_title('Training and validation accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    return fig


def plot_train_validation_bis(epochs, train_loss_values, validation_loss_values,
                              train_acc_values, validation_acc_values):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(epochs, train_loss_values, color="green", label='Training loss')
    plt.plot(epochs, validation_loss_values, color="red", label='Validation loss')

    plt.plot(epochs, train_acc_values, color="blue", label='Training acc')
    plt.plot(epochs, validation_acc_values, color="violet", label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    return fig


def build_model(n_layers=1, n_neurons=30, input_shape=65,
                optimizer="adam", metrics="accuracy"):
    """
    Cette fonction sert à construire et compiler un modèle Keras avec les hyperparamètres donnés
    """
    model = Sequential()
    model.add(Dense(units=n_neurons, activation="relu", input_shape=(input_shape,)))
    if n_layers > 1:
        for layer in range(n_layers - 1):
            model.add(Dense(units=n_neurons, activation="relu"))
    # Ajout d'un layer output
    model.add(Dense(units=1, activation="sigmoid"))

    # Compile du modèle
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    return model


# =================================================================================================
#
# =================================================================================================

os.chdir("/Users/quyen/PycharmProjects/Meteo_australie")

k = 4
path_classification = f"./output/classification/knn_impute_{k}"
path_output = os.path.abspath(os.sep.join([path_classification, "DNN"]))
data_file = f"./output/classification/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"

os.makedirs(path_output, exist_ok=True)

X_train, X_test, y_train, y_test = load(data_file)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
classes = ["Rain" if c == "Yes" else "No Rain" for c in label_encoder.classes_]

n_var_expl = X_train.shape[1]

model_1 = build_model(n_layers=2, n_neurons=64, input_shape=n_var_expl,
                      optimizer="adam", metrics="accuracy")

# model_1 = build_model(n_layers=1, n_neurons=64, input_shape=n_var_expl,
#                      optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics="accuracy")


t1 = time.time()
result_training = model_1.fit(X_train, y_train_encoded,
                              epochs=50, batch_size=50,
                              validation_data=(X_test, y_test_encoded),
                              verbose=1)
t2 = time.time()
print("temps = ", t2 - t1)


result_dict = result_training.history
train_loss_values = result_dict['loss']
validation_loss_values = result_dict['val_loss']
epochs = range(1, len(train_loss_values) + 1)

train_acc_values = result_dict['accuracy']
validation_acc_values = result_dict['val_accuracy']

fig = plot_train_validation(epochs, train_loss_values, validation_loss_values,
                            train_acc_values, validation_acc_values)
plt.show()

result_dict = result_training.history

print(model_1.evaluate(X_test, y_test_encoded))

#
# def build_model_one_layer(n_neurons, learning_rate, metrics="accuracy", input_shape=65):
#     model = Sequential()
#     model.add(Dense(units=n_neurons, activation="relu", input_shape=(input_shape,)))
#     model.add(Dense(units=1, activation="sigmoid"))
#
#     # Compile du modèle
#     adam = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=adam, loss="binary_crossentropy", metrics=metrics)
#
#     return model
#
#
# early_stopping = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
# keras_cl = KerasClassifier(build_fn=build_model_one_layer, learning_rate=[0.001, 0.01],
#                            n_neurons=np.arange(10, 100), verbose=0)
#
# param_distribs = {
#     "n_neurons": np.arange(10, 100),
#     "learning_rate": [0.001, 0.01]
# }
#
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
# rnd_search_cv = RandomizedSearchCV(keras_cl, param_distribs, n_iter=10, cv=kfold)
#
# t1 = time.time()
# rnd_search_cv.fit(X_train, y_train_encoded,
#                   epochs=200, batch_size=32,
#                   validation_data=(X_test, y_test_encoded),
#                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
#                   )
# t2 = time.time()
# print("temps = ", t2 - t1)


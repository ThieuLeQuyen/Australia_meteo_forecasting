# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""
import time
import os
import numpy as np
import pandas as pd
from context.context import ctx
from src.modelisation.pre_processing import PreProcessing
from joblib import dump

print(os.getcwd())
ctx.cmd_start()

fichier_data = "./data/weatherAUS.csv"

df_data = pd.read_csv(fichier_data)

with_location = False

# cols_to_drop = ["Date", "RainTomorrow", "RainToday", 'WindDir9am', 'WindDir3pm', 'WindGustDir', "Location",
#                 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
#                 'Temp3pm'
#                 ]
cols_to_drop = ["Date", "RainTomorrow", "RainToday", 'WindDir9am', 'WindDir3pm', 'WindGustDir', "Location"]
# var_cible = "Rainfall"
# var_cible = "MaxTemp"
var_cible = "MinTemp"

for k in [4]:
    print(k)
    prep = PreProcessing(df_data)
    # paramètre pour KNNImputer
    if with_location:
        data = prep.get_train_test(y_col=var_cible,
                                   cols_to_drop=cols_to_drop,
                                   drop_all_nan=True,
                                   cols_to_dummies=None,
                                   pct_missing_row=None,
                                   n_neighbors=k, weights="uniform",
                                   test_size=0.2,
                                   random_state=42, is_stratify=False)
        dump(data, f"./output/regression/data_pre_processing_with_location_knn_imputed_{k}_{var_cible}.joblib")
    else:
        data = prep.get_train_test(y_col=var_cible,
                                   cols_to_drop=cols_to_drop,
                                   drop_all_nan=True,
                                   cols_to_dummies=None,
                                   pct_missing_row=None,
                                   n_neighbors=k, weights="uniform",
                                   test_size=0.2,
                                   random_state=42, is_stratify=False)

        dump(data, f"./output/regression/data_pre_processing_without_location_knn_imputed_{k}_{var_cible}.joblib")

# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""
import os

import pandas as pd
from joblib import dump

from context.context import ctx
from src.modelisation.pre_processing import PreProcessing

print(os.getcwd())
ctx.cmd_start()

fichier_data = "./data/weatherAUS.csv"

df_data = pd.read_csv(fichier_data)

path_output = "./output/classification"
os.makedirs(path_output, exist_ok=True)

with_location = True
return_original_data = True

if with_location:
    var_cat = ['WindDir9am', 'WindDir3pm', 'WindGustDir', "Location"]
else:
    var_cat = ['WindDir9am', 'WindDir3pm', 'WindGustDir']

for k in [4, 5, 6]:
    print(k)
    prep = PreProcessing(df_data)
    # paramètre pour KNNImputer
    if with_location:
        data = prep.get_train_test(y_col="RainTomorrow",
                                   cols_to_drop=["Date"],
                                   drop_all_nan=False,
                                   cols_to_dummies=var_cat,
                                   pct_missing_row=None,
                                   n_neighbors=k, weights="uniform",
                                   return_original_data=return_original_data,
                                   test_size=0.2,
                                   random_state=42, is_stratify=True)
        if return_original_data:
            dump(data, f"{path_output}/data_original_pre_processing_with_location_knn_imputed_{k}.joblib")
        else:
            dump(data, f"{path_output}/data_scaled_pre_processing_with_location_knn_imputed_{k}.joblib")
    else:
        data = prep.get_train_test(y_col="RainTomorrow",
                                   cols_to_drop=["Date", "Location"],
                                   drop_all_nan=False,
                                   cols_to_dummies=var_cat,
                                   pct_missing_row=None,
                                   n_neighbors=k, weights="uniform",
                                   return_original_data=return_original_data,
                                   test_size=0.2,
                                   random_state=42, is_stratify=True)
        if return_original_data:
            dump(data, f"{path_output}/data_original_pre_processing_without_location_knn_imputed_{k}.joblib")
        else:
            dump(data, f"{path_output}/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib")

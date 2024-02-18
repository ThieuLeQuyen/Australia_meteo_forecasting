"""
Created on Sun Oct  1 19:56:30 2023

@author: Sophie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

import warnings

# preprocess
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# optimisation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow.keras.callbacks import LearningRateScheduler

from imblearn.over_sampling import RandomOverSampler, SMOTE
# auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# metrics
from sklearn.metrics import make_scorer, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import mean_squared_error, mean_absolute_error

# classifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# NN
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

# timeseries
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm       
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# pour couleurs
import plotly.express as px
import plotly.colors as pc

# divers
from tqdm import tqdm

# explicativité
import shap


#warnings.filterwarnings("ignore", message="is_sparse is deprecated")
warnings.filterwarnings("ignore", category=UserWarning)

def custom_callback(params, model, X, y):
    model.fit(X, y)
    score = model.score(X, y)
    print(f"Essai des hyperparamètres : {params}")
    print(f"Score d'entraînement : {score:.4f}")


class ProjetAustralieModelisation:

    def __init__(self, data:pd.DataFrame):
        self.X=None
        self.y=None
        #self.data = data.dropna()
        self.data = data
        
        self.data.index = pd.to_datetime(self.data.index)
        
        #self.data = self.data.drop(columns=["SaisonCos"])
        
        # si SaisonCos existe, alors on la renomme en 4pi et on ajoute SaisonCos2pi
        if hasattr(self.data, "SaisonCos"):
            self.data = self.data.rename(columns={'SaisonCos':'SaisonCos4pi'})
            self.data["SaisonCos2pi"] = np.cos(2*np.pi*(self.data.index.day_of_year-1)/365)
            
        #self.data = self.data.drop(columns=["SaisonCos2pi", "SaisonCos4pi"])

        # s'il n'y a que mount ginini en climat 5, on degage
        if (self.data[self.data.Climat==5].Location.nunique()==1):
            self.data = self.data[self.data.Climat!=5]
            
        # palette
        palette_set1 = px.colors.qualitative.Set1
        self.palette=[]
        for i in range(7):
            self.palette.append(pc.unconvert_from_RGB_255(pc.unlabel_rgb(palette_set1[i])))
            
        # libelle des climats
        self.lib_climats = {0:"Côte Est", 1:"Nord", 2:"Centre", 3:"Sud-Est", 4:"Intermédiaire", 5:"Mount Ginini", 6:"Côte Sud"}
            

    def copie(self, source):
        self.X = source.X
        self.y = source.y
        self.data = source.data
    
    # ---------
    # ---------
    # prepare les donnees pour l'apprentissage (2eme fonction la plus importante du module)
    # ---------
    # ---------
    
    def _modelisation_preparation(self, cible:str, scale:bool, climat:int=None, location:str="", cut2016:bool=False):
        
        # filtrage eventuel
        data=self.data
        if climat!=None:
            data = self.data[self.data.Climat==climat]
        if location!="":
            data = self.data[self.data.Location==location]

        # si cut2016 est a true, alors on va utiliser les données de 2016 comme données de validation et laisser les données anterieures pour l'apprentissage et les tests
#        if cut2016:
#            data["Date"]=date.index

            
        #self.X = data.drop(columns=cible)      
        self.Xy = data.copy()
        self.y = data[cible]
        
        # supprime toutes les infos sur la meteo future
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("Rain_J_")]
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("MaxTemp_J_")]
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("Rainfall_J_")]
               
        # eclate les climats (variable categorielle!)
        self.Xy = pd.get_dummies(self.Xy, columns=['Climat'])
#        self.Xy = pd.get_dummies(self.Xy, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])
        
        
        # supprime les autres colonnes donnant trop d'indices
        if cible=="RainToday":
            self.Xy = self.Xy.drop(columns=["Rainfall"]) # si on veut predire RainToday, on supprime Rainfall, sinon c'est de la triche...
        if cible.startswith("Rain_J_"):
            self.Xy = self.Xy.drop(columns=["RainTomorrow"]) # si on veut predire RainToday, on supprime Rainfall, sinon c'est de la triche...

        # enleve toutes les features enginerees precedemment
        #self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("Climat")]
        #self.Xy = self.Xy.drop(columns=["lat", "lng", "AmplitudeTemp", "SaisonCos2pi", "SaisonCos4pi"])
        #self.Xy = self.Xy.drop(columns=["SaisonCos2pi"])


        #self.X = self.X.drop(columns=self.X.columns[self.X.columns.str.startswith('WindGustDir')])
        if cible.startswith("Wind"):
            self.Xy = self.Xy.drop(columns=self.Xy.columns[self.Xy.columns.str.startswith('Wind')])

        # on reinjecte la cible, on fait un dropna et on eclate entre X et y        
        self.Xy[cible] = self.y
        self.Xy = self.Xy.dropna()       

        if hasattr(self.Xy, "RainTomorrow"):
            self.Xy.RainTomorrow = self.Xy.RainTomorrow.astype(int)
        if hasattr(self.Xy, "RainToday"):
            self.Xy.RainToday = self.Xy.RainToday.astype(int)       
        
        self.y = self.Xy[cible]
        self.X = self.Xy.drop(columns=cible)      
        
        # variable cible aleatoire
        ratio = len(data[data.RainTomorrow==0]) / len(data)
        #self.y = np.random.choice([0, 1], size=len(data), p=[ratio, 1-ratio])

        # si cut2016 est a true, alors on va utiliser les données de 2016 comme données de validation et laisser les données anterieures pour l'apprentissage et les tests
        if cut2016:
            self.X_2016=self.X[self.X.index>='2016-01-01']
            self.y_2016=self.y[self.y.index>='2016-01-01']
            self.X_2016_Location = self.X_2016.Location
            self.X_2016 = self.X_2016.drop(columns="Location")

            self.X = self.X[self.X.index<'2016-01-01']
            self.y=self.y[self.y.index<'2016-01-01']
            self.Xy = self.Xy[self.Xy.index<'2016-01-01']

        est_classification = cible.startswith("Rain") and not cible.startswith("Rainfall")
        if est_classification:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=66, stratify=self.y) 
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=66) 

        # on supprime la Location si elle est presente en str (utile uniquement pour filtrer en amont)
        #if hasattr(self.Xy, "Location"):
        self.X_Location = self.X.Location # la conservation permet d'indexer facilement plus tard selon les Location
        self.X_train_Location = X_train.Location # la conservation permet d'indexer facilement plus tard selon les Location
        self.X_test_Location = X_test.Location # la conservation permet d'indexer facilement plus tard selon les Location
        
        self.Xy = self.Xy.drop(columns="Location")
        self.X = self.X.drop(columns="Location")
        X_train = X_train.drop(columns="Location")
        X_test = X_test.drop(columns="Location")
        
        # normalisation
        self.scaler = None
        if scale:
            self.scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
            colX = self.X.columns
            self.X = pd.DataFrame(self.scaler.transform(self.X), columns=colX)

            # si cut2016 est a true, alors on va utiliser les données de 2016 comme données de validation et laisser les données anterieures pour l'apprentissage et les tests
            if cut2016:
                self.X_2016 = self.scaler.transform(self.X_2016)

            
        oversample = RandomOverSampler()
        # pip install threadpoolctl==3.1.0  pour avoir SMOTE sur + de 15 colonnes
        #oversample = SMOTE()
        #X_train, y_train = oversample.fit_resample(X_train, y_train)          
            
            
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    # prepare les donnees en vue d'entrainer un RNN
    # pre-requis: preparation sur une location donnée uniquement (pour garantir unicité des dates et avoir une coherence temporelle) et ne pas se disperser sur differents lieux chaque jour
    def _modelisation_preparation_rnn(self, location:str, cible:str):
        
        # longueur des sequences
#        self.rnn_sequence_longueur = 15
#        self.rnn_batch_size = 1

        #self.rnn_sequence_longueur = 10
        self.rnn_sequence_longueur = 15
        
        self.rnn_batch_size = 1
        
        # la cible n'a en realite pas d'importance ici
        self._modelisation_preparation(cible=cible, scale=False, location=location)
        
        self.XyOrig = self.Xy[cible]
        
        # comme on est sur une location en particulier, pas d'interet de conserver les climats
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("Climat_")]
        # lat et lng sont identiques puisque meme Location: on droppe aussi
        # raintomorrow n'a pas non plus d'interet dans cette approche
        self.Xy = self.Xy.drop(columns=["lat", "lng"])#, "RainTomorrow"])
        
        
        self.scaler_rnn = preprocessing.MinMaxScaler().fit(self.Xy)
        colXy = self.Xy.columns
        indexXy = self.Xy.index
        self.Xy = pd.DataFrame(self.scaler_rnn.transform(self.Xy), columns=colXy, index=indexXy)
        
        proportion_1=.7
        proportion_2=.9
        self.indice_coupure_1 = int(len(self.Xy) * proportion_1)
        self.indice_coupure_2 = int(len(self.Xy) * proportion_2)

        if self.rnn_multivariee:
            # multivarie
            self.X_train = self.Xy[:self.indice_coupure_1].to_numpy()
            self.X_validation = self.Xy[self.indice_coupure_1:self.indice_coupure_2].to_numpy()
            self.X_test = self.Xy[self.indice_coupure_2:].to_numpy()
        else:   
            # monovarie
            self.X_train = self.Xy[cible][:self.indice_coupure_1].to_numpy()
            self.X_validation = self.Xy[cible][self.indice_coupure_1:self.indice_coupure_2].to_numpy()
            self.X_test = self.Xy[cible][self.indice_coupure_2:].to_numpy()

        self.y_train = self.X_train
        self.y_validation = self.X_validation
        self.y_test = self.X_test
        
        
        self.rnn_train_generator = TimeseriesGenerator(self.X_train, 
                                              self.y_train,
                                              length = self.rnn_sequence_longueur,
                                              batch_size=self.rnn_batch_size,
                                              stride=1
                                              )
        
        self.rnn_validation_generator = TimeseriesGenerator(self.X_validation, 
                                             self.y_validation,
                                             length = self.rnn_sequence_longueur,
                                             batch_size=self.rnn_batch_size,
                                             stride=1
                                             )

        self.rnn_test_generator = TimeseriesGenerator(self.X_test, 
                                             self.y_test,
                                             length = self.rnn_sequence_longueur,
                                             batch_size=self.rnn_batch_size,
                                             stride=1
                                             )

        """
        self._cree_sequence_rnn()
        
        proportion=.6
        indice_coupure = int(len(self.rnn_cibles) * proportion)
        self.X_train = self.rnn_features[:indice_coupure]
        self.y_train = self.rnn_cibles[:indice_coupure]
        self.X_test = self.rnn_features[indice_coupure:]
        self.y_test = self.rnn_cibles[indice_coupure:]      
        """
        
    # cree une sequence pour le rnn
    def _cree_sequence_rnn(self):
        # on va creer de nombreuses fenetres de predictions       
        
        rnn_features = []
        rnn_cibles = []
        
        # on va deplacer une fenetre du debut à la fin des dates
        for i in range(len(self.Xy)-self.rnn_sequence_longueur):
        #for i in range(3):
            
            sequence = self.Xy.iloc[i:i+self.rnn_sequence_longueur]
            features = sequence.iloc[:-1].values
            cible = sequence.iloc[-1].values
            
            rnn_features.append(features)
            rnn_cibles.append(cible)
        
        self.rnn_features = rnn_features
        self.rnn_cibles = rnn_cibles           
        
    # cree un RNN de regression
    def modelisation_rnn_reg(self, location:str, cible:str, multivariee:bool=False):
        print (time.ctime())      
        
        # analyse mono ou multivariée
        self.rnn_multivariee = multivariee

        print (f'\n -------\nModelisation de {cible} avec un RNN\n -------\n')
        i_temps_debut=time.time()
        
        self._modelisation_preparation_rnn(location, cible)

        modele = Sequential()

        num_features = 1
        if len(self.rnn_train_generator.data.shape)==1:
            num_features=1
        else:
            num_features = self.rnn_train_generator.data.shape[1]

# v0
#        modele.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(self.rnn_sequence_longueur, num_features)))
#        modele.add(LSTM(10, activation='relu'))

# v1
        if multivariee:
            # réseau +  complexe si multivarié
            modele.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(self.rnn_sequence_longueur, num_features)))
            modele.add(LSTM(100, activation='relu'))
            modele.add(Dense(100, activation='relu'))            

        else:
            modele.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(self.rnn_sequence_longueur, num_features)))
    #        modele.add(Dropout(.1))
            modele.add(LSTM(10, activation='relu'))
            #modele.add(Dense(100, activation='relu'))                        
    #        modele.add(Dropout(.1))       
            #modele.add(Dense(100, activation='relu'))
        
        # nb de neurones en sortie = nb de feature à prédire pour alimenter à nouveau le modele pour les predictions des jours futurs
#        modele.add(Dense(self.Xy.shape[1]))
        modele.add(Dense(num_features))
        
        opt = Adam(lr=1e-4)
        modele.compile(optimizer=opt, loss='mse')
        
        print (modele.summary())
        #modele.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=opt)

        #history = modele.fit(self.X_train, self.y_train, epochs=10, batch_size=128, validation_split=.2, verbose=1)
#        history = modele.fit_generator(generator=self.rnn_train_generator, epochs=80, verbose=1, validation_data=self.rnn_validation_generator)
        history = modele.fit_generator(generator=self.rnn_train_generator, epochs=30, verbose=1, validation_data=self.rnn_validation_generator)

        self.modele=modele
        self.history=history

        print (time.ctime())
        
        self.resultats_rnn(location, cible, multivariee)

    # cree un RNN de classification
    def modelisation_rnn_clf(self, location:str, cible:str, multivariee:bool=False):
        print (time.ctime())      

        # analyse mono ou multivariée
        self.rnn_multivariee = multivariee

        print (f'\n -------\nModelisation de {cible} avec un RNN\n -------\n')
        i_temps_debut=time.time()
        
        self._modelisation_preparation_rnn(location, cible)

        modele = Sequential()

        num_features = 1
        if len(self.rnn_train_generator.data.shape)==1:
            num_features=1
        else:
            num_features = self.rnn_train_generator.data.shape[1]


#        modele.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(self.rnn_sequence_longueur, self.Xy.shape[1])))
#        modele.add(LSTM(30, activation='tanh', return_sequences=True, input_shape=(self.rnn_sequence_longueur, num_features)))

        if multivariee:
            # on a un reseau + complexe pour le multivarie
            modele.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(self.rnn_sequence_longueur, num_features)))
            #modele.add(Dropout(.1))
    #        modele.add(LSTM(10, activation='tanh'))
            modele.add(LSTM(50, activation='relu'))        
    #        modele.add(Dense(self.Xy.shape[1]))
    #        modele.add(Dense(50))
            modele.add(Dense(50))
        
        else:
            modele.add(LSTM(5, activation='tanh', return_sequences=True, input_shape=(self.rnn_sequence_longueur, num_features)))
            modele.add(LSTM(10, activation='relu'))        
            modele.add(Dense(10))
            
        
        modele.add(Dense(num_features, activation='sigmoid'))
        opt = Adam(lr=1e-3)
        #modele.compile(optimizer=opt, loss='mse')
        
        # en multivarie, il y a des variables continues et des variables binaires => il nous faut deux loss
        if multivariee:
            modele.compile(loss=['binary_crossentropy', 'mse'], metrics=['accuracy'], optimizer=opt)
        # en monovarie, on est juste sur une classification binaire
        else:
            modele.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
        
        print (modele.summary())

        #history = modele.fit(self.X_train, self.y_train, epochs=10, batch_size=128, validation_split=.2, verbose=1)
        history = modele.fit_generator(generator=self.rnn_train_generator, epochs=10, verbose=1, validation_data=self.rnn_validation_generator)

        self.modele=modele
        self.history=history

        print (time.ctime())
        
        self.resultats_rnn_clf(location, cible, multivariee)
        
    # affiche les resultats du RNN CLF entrainé
    def resultats_rnn_clf(self, location:str="", cible:str="RainTomorrow", multivariee:bool=False):
        
        str_multivarie="monovarié"
        if multivariee:
            str_multivarie="multivarié"            
        nom_modele = self.titre_graphe("RNN "+str_multivarie, "", location=location, cible=cible)
        
        plt.figure(figsize=(16, 6))
        plt.plot(self.history.history['val_accuracy'], "g", label="Accuracy (Val)")
        plt.plot(self.history.history['accuracy'], "b", label="Accuracy (Train)")
        plt.plot(self.history.history['loss'], "r", label="Loss")
        #plt.ylim((.2,.85))
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Historique d'Accuracy - \n{nom_modele}")
        plt.show();

        minmax = MinMaxScaler()

        train_pred = self.modele.predict(self.rnn_train_generator)
        train_pred = minmax.fit_transform(train_pred)

        y_reel = self.y_train[self.rnn_sequence_longueur:]

        if not multivariee:
            y_pred_proba = train_pred.reshape(-1)
        else:
            y_pred_proba = train_pred
        y_pred = y_pred_proba>.5

        self.aa0=y_reel
        self.bb0=y_pred

        if multivariee:
            y_pred = pd.DataFrame(y_pred, columns=self.Xy.columns)[cible]
            y_pred_proba = pd.DataFrame(y_pred_proba, columns=self.Xy.columns)[cible]
            y_reel = pd.DataFrame(y_reel, columns=self.Xy.columns)[cible]
            
        
        self.aa=y_reel
        self.bb=y_pred
        
        print(classification_report(y_reel, y_pred))
        print (confusion_matrix(y_reel, y_pred))

        #print(self.modele.evaluate(self.X_test, self.y_test))
        
        self.trace_courbe_roc_ann(y_reel, y_pred_proba, nom_modele)
        
        print ("Seuil par défaut:")
        self.scores_classification(y_reel, y_pred)
        #print ("\nSeuil optimal:")
        #self.scores_classification(self.y_test, self.test_pred.reshape(-1)>=self.res_roc_best_seuil)
        
        print (time.ctime())
        
    # affiche les resultats du RNN entrainé
    def resultats_rnn(self, location:str, cible:str, multivariee:bool=False):
        train_pred = self.modele.predict(self.rnn_train_generator)
        
        if not self.rnn_multivariee:
            train_pred = np.repeat(train_pred, self.Xy.shape[1], axis=-1)    
        self.train_pred_unscaled = pd.DataFrame(self.scaler_rnn.inverse_transform(train_pred), columns=self.Xy.columns)[cible]
        self.train_pred_unscaled.index = self.Xy[self.rnn_sequence_longueur:self.indice_coupure_1].index

        validation_pred = self.modele.predict(self.rnn_validation_generator)
        if not self.rnn_multivariee:
            validation_pred = np.repeat(validation_pred, self.Xy.shape[1], axis=-1)
        self.validation_pred_unscaled = pd.DataFrame(self.scaler_rnn.inverse_transform(validation_pred), columns=self.Xy.columns)[cible]
        self.validation_pred_unscaled.index = self.Xy[self.indice_coupure_1+self.rnn_sequence_longueur:self.indice_coupure_2].index

        train_orig_unscaled = pd.DataFrame(columns=self.Xy.columns).astype(self.Xy.dtypes.to_dict())
        if not multivariee:
            #train_orig_unscaled[cible] = self.X_train.iloc[self.rnn_sequence_longueur:]
            train_orig_unscaled[cible] = self.X_train[self.rnn_sequence_longueur:]
        else:
            train_orig_unscaled[cible] = pd.DataFrame(self.X_train[self.rnn_sequence_longueur:], columns=self.Xy.columns)[cible]
            
        self.train_orig_unscaled = pd.DataFrame(self.scaler_rnn.inverse_transform(train_orig_unscaled), columns=self.Xy.columns)[cible]
        self.train_orig_unscaled.index = self.Xy[self.rnn_sequence_longueur:self.indice_coupure_1].index

        validation_orig_unscaled = pd.DataFrame(columns=self.Xy.columns).astype(self.Xy.dtypes.to_dict())
        if not multivariee:
            #validation_orig_unscaled[cible] = self.X_validation.iloc[self.rnn_sequence_longueur:]
            validation_orig_unscaled[cible] = self.X_validation[self.rnn_sequence_longueur:]
        else:
            validation_orig_unscaled[cible] = pd.DataFrame(self.X_validation[self.rnn_sequence_longueur:], columns=self.Xy.columns)[cible]
        self.validation_orig_unscaled = pd.DataFrame(self.scaler_rnn.inverse_transform(validation_orig_unscaled), columns=self.Xy.columns)[cible]
        self.validation_orig_unscaled.index = self.Xy[self.indice_coupure_1+self.rnn_sequence_longueur:self.indice_coupure_2].index

        plt.figure(figsize=(50, 6))
#        plt.figure(figsize=(30, 6))
        plt.plot(self.train_orig_unscaled, label="Train Orig", alpha=.75)
        plt.plot(self.train_pred_unscaled, label="Train Pred", alpha=.75)
        plt.plot(self.validation_orig_unscaled, label="Val Orig", alpha=.75)
        plt.plot(self.validation_pred_unscaled, label="Val Pred", alpha=.75)
        plt.legend()
        plt.show();


        print ("RMSE Train: ",np.sqrt(mean_squared_error(self.train_orig_unscaled, self.train_pred_unscaled)))
        print ("RMSE Valid: ",np.sqrt(mean_squared_error(self.validation_orig_unscaled, self.validation_pred_unscaled)))
        
        print ("MAE Train: ",mean_absolute_error(self.train_orig_unscaled, self.train_pred_unscaled))
        print ("MAE Valid: ",mean_absolute_error(self.validation_orig_unscaled, self.validation_pred_unscaled))
        
#        test_pred = np.repeat(test_pred, self.Xy.shape[1], axis=-1)       
#        test_pred = modele.predict(self.rnn_test_generator)
        
        nom_modele = self.titre_graphe(nom_modele="RNN", location=location, cible=cible, hp="")
        
        plt.figure(figsize=(16, 6))
        plt.plot(self.history.history['val_loss'], "g", label="MSE (Val)")
        plt.plot(self.history.history['loss'], "b", label="MSE (Train)")
        #plt.ylim((.5,.85))
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title(nom_modele)
        plt.show();
        """        
        self.test_pred = modele.predict(self.X_test)
        self.test_pred_class = self.test_pred>.5
        
        print(classification_report(self.y_test, self.test_pred_class))
        print (confusion_matrix(self.y_test, self.test_pred_class))

        print(modele.evaluate(self.X_test, self.y_test))

        i_fin_completes=time.time()
        print (" Temps comp: {:.2f} minutes".format( (i_fin_completes-i_temps_debut)/60))
        
        self.trace_courbe_roc_dnn(self.test_pred, nom_modele)
        """        
        print (time.ctime())
        
    def prediction2016_rnn(self, cible:str="MaxTemp", nb_prev:int=30):
        if self.rnn_batch_size>1:
            print ("Il faut que le batch_size soit à 1 pour pouvoir faire la prediction increntale")
            return
        
        pred =[]
        reel = []
        
        X = self.rnn_test_generator[0][0]
        y = self.rnn_test_generator[0][1][0]
        y_pred = self.modele.predict(X)[0][0]
        pred.append(y_pred)
        reel.append(y)
        
        for i in range(nb_prev):
            # decale d'un jour la fenetre en ajoutant la derniere prediction
            X = X[0][1:]
            X = np.append(X,y_pred)
            X = X.reshape(1,-1)
            y_pred = self.modele.predict(X)[0][0]
            pred.append(y_pred)
            y_reel = self.rnn_test_generator[i+1][1][0]
            reel.append(y_reel)
            print (i, y_pred, y_reel, X)
            
        self.rnn_pred2016_pred = pred
        self.rnn_pred2016_orig = reel
        
        pred_xtend = pred
        if not self.rnn_multivariee:       
            pred_xtend = np.repeat(np.array(pred).reshape(len(pred),1), self.Xy.shape[1], axis=-1)
        pred_unscaled = pd.DataFrame(self.scaler_rnn.inverse_transform(pred_xtend), columns=self.Xy.columns)[cible]
        pred_unscaled.index = self.Xy[self.indice_coupure_2+self.rnn_sequence_longueur:self.indice_coupure_2+self.rnn_sequence_longueur+nb_prev+1].index
        
        reel_xtend = reel
        if not self.rnn_multivariee:       
            reel_xtend = np.repeat(np.array(reel).reshape(len(reel),1), self.Xy.shape[1], axis=-1)
        reel_unscaled = pd.DataFrame(self.scaler_rnn.inverse_transform(reel_xtend), columns=self.Xy.columns)[cible]
        reel_unscaled.index = self.Xy[self.indice_coupure_2+self.rnn_sequence_longueur:self.indice_coupure_2+self.rnn_sequence_longueur+nb_prev+1].index
        
        plt.figure(figsize=(16, 6))
        plt.plot(pred_unscaled, label="Predictions incrémentales")
        plt.plot(reel_unscaled, label="Donnees reelles")
        plt.legend()
        plt.title("Prediction des températures sur période non vue à l'entraînement")
        plt.show();
        
        print ("RMSE test: ",np.sqrt(mean_squared_error(reel_unscaled, pred_unscaled)))
        print ("MAE test: ",mean_absolute_error(reel_unscaled, pred_unscaled))               
    
    # --------
    #  entraine des modeles sur 365 jours de prediction de Rain_J
    # --------   

    # determine les pvalue pour chaque jour de prediction dans le futur afin de determiner le nb de journées possibles de predictions pour toute l'australie
    def entraine_rainj_macro(self):
        if not hasattr(self, "resultats_rainj_location"):
            self.resultats_rainj_macro = pd.DataFrame(columns=["J", "seuil", "pvalue05", "pvaluebest", "AUC", "AccuracyTrain", "AccuracyTest", "RecallTest"])
        
        for i in range(1,365):
            self.modelisation(cible=f"Rain_J_{i:02d}", gs=True)
            pv_05, pv_best = self.verification_significativite_modele()
            self.resultats_rainj_macro.loc[len(self.resultats_rainj_macro)] = [i, 1, pv_05, pv_best, self.res_roc_auc, self.res_acc_train_seuil, self.res_acc_test_seuil, self.res_recall_test_seuil]
            self.resultats_rainj_macro.loc[len(self.resultats_rainj_macro)] = [i, 0, pv_05, pv_best, self.res_roc_auc, self.res_acc_train, self.res_acc_test, self.res_recall_test]
        
    # determine les pvalue pour chaque jour de prediction dans le futur afin de determiner le nb de journées possibles de predictions pour une location donnee
    def entraine_rainj_location(self, locations:list):
        if not hasattr(self, "resultats_rainj_location"):
            self.resultats_rainj_location = pd.DataFrame(columns=["J", "Location", "pvalue05", "pvaluebest", "AUC", "AccuracyTrain", "AccuracyTest", "RecallTest"])
        
        for location in locations:
            for i in range(1,365):
                self.modelisation(cible=f"Rain_J_{i:02d}", location=location, gs=True)
                pv_05, pv_best = self.verification_significativite_modele()
                self.resultats_rainj_location.loc[len(self.resultats_rainj_location)] = [i, location, pv_05, pv_best, self.res_roc_auc, self.res_acc_train_seuil, self.res_acc_test_seuil, self.res_recall_test_seuil]

    # determine les pvalue pour chaque jour de prediction dans le futur afin de determiner le nb de journées possibles de predictions pour un climat donné
    def entraine_rainj_climat(self, climat:int):
        if not hasattr(self, "resultats_rainj_climat"):
            self.resultats_rainj_climat = pd.DataFrame(columns=["J", "Climat", "pvalue05", "pvaluebest", "AUC", "AccuracyTrain", "AccuracyTest", "RecallTest"])
        
        if not hasattr(self, "resultats_rainj_climat_location"):
            self.resultats_rainj_climat_location = pd.DataFrame(columns=["J", "Location", "pvalue05", "pvaluebest", "AUC", "AccuracyTrain", "AccuracyTest", "RecallTest"])
        liste_locations = self.data[self.data.Climat==climat].Location.unique()
        
        for i in range(1,365):
            self.modelisation(cible=f"Rain_J_{i:02d}", climat=climat, gs=True)
            pv_05, pv_best = self.verification_significativite_modele()
            self.resultats_rainj_climat.loc[len(self.resultats_rainj_climat)] = [i, climat, pv_05, pv_best, self.res_roc_auc, self.res_acc_train_seuil, self.res_acc_test_seuil, self.res_recall_test_seuil]
            
            for location in liste_locations:
                self.performance_climat_sur_location(location,i)

    # à partir d'un modele par climat, determine les performances sur une location
    def performance_climat_sur_location(self, location:str, J:int):
        X_train_Loc = self.X_train[self.X_train_Location == location]
        y_train_Loc = self.y_train[self.X_train_Location == location]
        X_test_Loc = self.X_test[self.X_test_Location == location]
        y_test_Loc = self.y_test[self.X_test_Location == location]
        
        y_train_pred_seuil = self.modele.predict_proba(X_train_Loc)[:,1] >= self.res_roc_best_seuil
        y_test_pred_seuil = self.modele.predict_proba(X_test_Loc)[:,1] >= self.res_roc_best_seuil
        y_test_pred = self.modele.predict(X_test_Loc)
               
        res_acc_train_seuil = accuracy_score(y_train_Loc, y_train_pred_seuil)
        res_acc_test_seuil = accuracy_score(y_test_Loc, y_test_pred_seuil)
        res_recall_test_seuil = recall_score(y_test_Loc, y_test_pred_seuil)
        
        pv_05 = self.verification_significativite(y_test_Loc, y_test_pred)
        pv_best = self.verification_significativite(y_test_Loc, y_test_pred_seuil)
        
        self.resultats_rainj_climat_location.loc[len(self.resultats_rainj_climat_location)] = [J, location, pv_05, pv_best, 2, res_acc_train_seuil, res_acc_test_seuil, res_recall_test_seuil]                 
        
    # enregistre dans un DF les performances du modele
    def performance_modele(self, climat:int="", location:str=""):
        if not hasattr(self, "resultats_modeles"):
            self.resultats_modeles = pd.DataFrame(columns=["Climat", "Location", "Seuil", "pvalue05", "pvaluebest", "AccuracyTrain", "AccuracyTest", "RecallTest", "PrecisionTest", "F1Test", "AUC"])
        pv_05, pv_best = self.verification_significativite_modele()
        self.resultats_modeles.loc[len(self.resultats_modeles)] = [climat, location, self.res_roc_best_seuil, pv_05, pv_best, self.res_acc_train_seuil, self.res_acc_test_seuil, self.res_recall_test_seuil, self.res_precision_test_seuil, self.res_f1_test_seuil, self.res_roc_auc]
        self.resultats_modeles.loc[len(self.resultats_modeles)] = [climat, location, .5, pv_05, pv_best, self.res_acc_train, self.res_acc_test, self.res_recall_test, self.res_precision_test, self.res_f1_test, self.res_roc_auc]
        
    def performance_modeles(self, modele_micro:bool=False, modele_climat:bool=False):
        if not hasattr(self, "resultats_modeles"):
            self.resultats_modeles = pd.DataFrame(columns=["Climat", "Location", "Seuil", "pvalue05", "pvaluebest", "AccuracyTrain", "AccuracyTest", "RecallTest", "PrecisionTest", "F1Test", "AUC"])

        if modele_micro:
            for location in self.data.Location.unique():
                print ("\n\nLocation ", location,"\n\n")
                self.modelisation(location=location, gs=True)
                self.performance_modele("", location)
                self.affiche_feature_importance(f"{location}")

        if modele_climat:
            for climat in self.data.Climat.unique():
                print ("\n\nClimat ", climat,"\n\n")
                self.modelisation(climat=climat, gs=True)
                self.performance_modele(climat, "")
                self.affiche_feature_importance(f"climat {climat} ({self.lib_climats[climat]})")
       
    # applique modele actuel sur une location donnée
    def applique_sur_location(self, location:str):
        xt=self.X_train[self.X_train_Location==location]
        yt=self.y_train[self.X_train_Location==location]
        y_predt = self.modele.predict(xt)
        print("Acc train", accuracy_score(yt, y_predt))
        
        
        x=self.X_test[self.X_test_Location==location]
        y=self.y_test[self.X_test_Location==location]
        y_pred = self.modele.predict(x)

        y_test_pred_proba = self.modele.predict_proba(x)
        fpr, tpr, thresholds = roc_curve(y, y_test_pred_proba[:,1])
        roc_auc = auc(fpr, tpr)
        print("AUC ", roc_auc)
        
        # scores            
        print ("Seuil par défaut:")
        self.scores_classification(y, y_test_pred_proba[:,1]>=.5)
#        print ("\nSeuil optimal:")
#        self.scores_classification(y, y_test_pred_proba[:,1]>=self.res_roc_best_seuil)
        
        
        self.affiche_feature_importance()
       
    # affiche feature importance
    def affiche_feature_importance(self, libelle:str=""):
        # affiche feature importance
        plt.figure(figsize=(6,8))
        sorted_idx = self.modele.feature_importances_.argsort()
        plt.barh(self.X.columns[sorted_idx], self.modele.feature_importances_[sorted_idx])
        
        lib = "Feature Importance"
        if libelle!="":
           lib +=" pour "+libelle+" ("+self.cible+")"
        plt.xlabel(lib)
        plt.show();
        
    # entraine toutes les zones climatiques
    def entraine_rainj_tous_climats(self):
        self.resultats_rainj_climat = pd.DataFrame(columns=["J", "Climat", "pvalue05", "pvaluebest", "AUC", "AccuracyTrain", "AccuracyTest", "RecallTest"])
        self.resultats_rainj_climat_location = pd.DataFrame(columns=["J", "Location", "pvalue05", "pvaluebest", "AUC", "AccuracyTrain", "AccuracyTest", "RecallTest"])
        for i in np.sort(self.data.Climat.unique()):
            self.entraine_rainj_climat(i)
        
        
    # predictions sur l'année 2016 pour une location
    def predictions2016(self, location:str, climat:int, modele_micro:bool=False, modele_climat:bool=False):
        liste_y = []
        liste_y_pred = []
        liste_y_pred_seuil = []             
        liste_y_pred_proba = []
        liste_roc_best_seuil = []
        
        dates = pd.date_range(start='2016-01-04', end='2016-12-31', freq='D')        
        j_decalage = 3 # pour partir un peu plus loin que le 1er janvier (penser à maj le libellé ci-dessus)
        
        for i in range(360):
            if modele_micro:
                self.modelisation(cible=f"Rain_J_{i+1:02d}", location=location, gs=True, cut2016=True)            
            elif modele_climat:
                self.modelisation(cible=f"Rain_J_{i+1:02d}", climat=climat, gs=True, cut2016=True)            
            else:
                self.modelisation(cible=f"Rain_J_{i+1:02d}", gs=True, cut2016=True)            

            X0 = self.X_2016[self.X_2016_Location == location][j_decalage]
            y0 = self.y_2016[self.X_2016_Location == location][j_decalage]
            liste_y.append(y0)

            y_pred_proba = self.modele.predict_proba(pd.DataFrame(X0).T)[:,1][0] # on predit chaque fois à partir de la meme date
            liste_y_pred_proba.append(y_pred_proba)
            liste_y_pred.append((y_pred_proba >= .5))
            liste_y_pred_seuil.append((y_pred_proba >= self.res_roc_best_seuil))
            
            liste_roc_best_seuil.append(self.res_roc_best_seuil)
            
            self.liste_y = liste_y
            self.liste_y_pred = liste_y_pred
            self.liste_y_pred_seuil = liste_y_pred_seuil
            self.liste_y_pred_proba = liste_y_pred_proba
            
            
        print(confusion_matrix(liste_y, liste_y_pred))
        conf_seuil = confusion_matrix(liste_y, liste_y_pred_seuil)
        print(conf_seuil)

        print ("Accuracy:")
        acc_seuil = accuracy_score(liste_y, liste_y_pred_seuil)
        print(accuracy_score(liste_y, liste_y_pred))
        print(acc_seuil)

        print ("Recall:")
        rec_seuil = recall_score(liste_y, liste_y_pred_seuil)
        print(recall_score(liste_y, liste_y_pred))
        print(rec_seuil)
        
        print ("X²:")
        x2_seuil = self.verification_significativite(liste_y, liste_y_pred_seuil)
        print (self.verification_significativite(liste_y, liste_y_pred))
        print (x2_seuil)
            
        plt.figure(figsize=(20,2))
        plt.plot(dates.values[0:len(liste_y)], liste_y, label="Reel", color="#00F")
        plt.plot(dates.values[0:len(liste_y)], liste_y_pred_seuil, label="Predictions", color="#0A0")
        #plt.plot(liste_y_pred, label="Predictions standard", color="#070")
        plt.plot(dates.values[0:len(liste_y)], liste_y_pred_proba, label="Probablité de pluie", color="#F40")
        
        plt.plot(dates.values[0:len(liste_y)], liste_roc_best_seuil, label="Seuil optimal", color="#BBB", linestyle="--")
        
        #plt.axhline(self.res_roc_best_seuil, color = '#AAA', linestyle = '--')
        plt.legend()
        plt.title(f"{location}, climat {climat} ({self.lib_climats[climat]}) - p-value X²: {x2_seuil:.3f} - Taux de jours de pluies : {(np.sum(liste_y)/len(liste_y)):.3f} - Accuracy: {acc_seuil:.3f} - Recall: {rec_seuil:.3f}\nMatrice de confusion:\n {conf_seuil}")
        plt.show();
        
        
    
    # predictions sur l'année 2016 pour une location
    # nope: là, on predit Rain_J pour toute l'année 2016. ce n'est pas ce que je veux faire
    def nope_predictions2016(self, location:str):
        X = self.X_2016[self.X_2016_Location == location]
        y = self.y_2016[self.X_2016_Location == location]

        y_pred_proba = self.modele.predict_proba(X)
        y_pred = y_pred_proba[:,1] >= .5
        y_pred_seuil = y_pred_proba[:,1] >= self.res_roc_best_seuil
        
        plt.figure(figsize=(40,2))
        plt.plot(y, label="Reel", color="#F00")
        plt.plot(y_pred_seuil, label="Predictions seuil optimal", color="#0F0")
        plt.plot(y_pred, label="Predictions standard", color="#070")
        plt.legend()
        plt.show();
        
        print(confusion_matrix(y, y_pred))
        print(confusion_matrix(y, y_pred_seuil))

        print(accuracy_score(y, y_pred))
        print(accuracy_score(y, y_pred_seuil))

        print(recall_score(y, y_pred))
        print(recall_score(y, y_pred_seuil))

        
    # --------------
    #    affichage de resultats
    # ---------------
    
    # affiche les variables de performances pour toute l'australie
    def affiche_pvalue_rainj_macro(self, nbj=365):
        self.resultats_rainj_macro = pd.read_csv("resultats_rainj_macro.csv")
        
        figure = plt.figure(figsize=(12,4))
        xtick = [*range(1,16), *range(16,nbj,7)]

        masque = (self.resultats_rainj_macro.seuil>.1) & (self.resultats_rainj_macro.J<=nbj)
        
        plt.plot(self.resultats_rainj_macro[masque].J.values, self.resultats_rainj_macro[masque].pvalue05.values, label="p-value d'après prédictions selon seuil par défaut (0,5)")
        plt.plot(self.resultats_rainj_macro[masque].J.values, self.resultats_rainj_macro[masque].pvaluebest.values, label="p-value d'après prédictions selon seuil optimal")
        plt.title("pvalues pour tout l'Australie")
        plt.axhline(.05, color = 'r', linestyle = '--')
        plt.xticks(xtick)
        plt.legend(loc="upper right")

        plt.show();

    # affiche les variables de performances pour toute l'australie
    def affiche_perfs_rainj_macro(self, nbj=365):
        self.resultats_rainj_macro = pd.read_csv("resultats_rainj_macro.csv")
        
        figure = plt.figure(figsize=(8,4))
        xtick = [*range(1,16), *range(16,nbj,7)]

        masque = (self.resultats_rainj_macro.seuil>.1) & (self.resultats_rainj_macro.J<=nbj)

        plt.plot(self.resultats_rainj_macro[masque].J.values, self.resultats_rainj_macro[masque].AccuracyTrain.values, label="Accuracy (train)")
        plt.plot(self.resultats_rainj_macro[masque].J.values, self.resultats_rainj_macro[masque].AccuracyTest.values, label="Accuracy (test)")
        plt.plot(self.resultats_rainj_macro[masque].J.values, self.resultats_rainj_macro[masque].RecallTest.values, label="Recall (test)")
        plt.plot(self.resultats_rainj_macro[masque].J.values, self.resultats_rainj_macro[masque].AUC.values, label="AUC")
        plt.title("Variables de Performances pour toute l'Australie\n(seuil optimal)")
        plt.axhline(.9, color = '#666', linestyle = '--')
        plt.axhline(.8, color = '#666', linestyle = '--')
        plt.axhline(.7, color = '#666', linestyle = '--')
        plt.axhline(.6, color = '#666', linestyle = '--')
        #ax[num_ax].axhline(.55, color = '#666', linestyle = '--')
        plt.axhline(.5, color = '#666', linestyle = '--')
        plt.xticks(xtick)
        plt.xlabel("Nb de journées de prédiction de la pluie dans le futur")
        plt.ylabel("Score")
      
        plt.legend(loc="upper right")
        
        plt.ylim(0,1)

        plt.show();
    
    
    # affiche les variables de performances pour une zone climatique
    def affiche_pvalue_rainj_climats(self):
        self.resultats_rainj_climat = pd.read_csv("resultats_rainj_climat.csv")
        
        figure, ax = plt.subplots(self.resultats_rainj_climat.Climat.nunique(), 1, figsize=(25,20))
        xtick = [*range(1,8), *range(8,365,7)]
        num_ax=0
        for i in np.sort(self.resultats_rainj_climat.Climat.unique()):
            
            ax[num_ax].plot(self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].J.values, self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].pvalue05.values, label="p-value d'après prédictions selon seuil par défaut (0,5)")
            ax[num_ax].plot(self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].J.values, self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].pvaluebest.values, label="p-value d'après prédictions selon seuil optimal")
            ax[num_ax].set_title(f"pvalues pour climat {i} ({self.lib_climats[i]})")
            ax[num_ax].axhline(.05, color = 'r', linestyle = '--')
            ax[num_ax].set_xticks(xtick)
            ax[num_ax].legend(loc="upper right")

            num_ax+=1
        plt.show();
        
    # affiche les variables de performances pour une zone climatique
    def affiche_perfs_rainj_climats(self):
        self.resultats_rainj_climat = pd.read_csv("resultats_rainj_climat.csv")
        
        figure, ax = plt.subplots(self.resultats_rainj_climat.Climat.nunique(), 1, figsize=(25,20))
        #figure, ax = plt.subplots(self.resultats_rainj_climat.Climat.nunique()+1, 1, figsize=(25,20))
        
        xtick = [*range(1,8), *range(8,365,7)]
        num_ax=0
        for i in np.sort(self.resultats_rainj_climat.Climat.unique()):
        #for i in []:
            ax[num_ax].plot(self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].J.values, self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].AccuracyTrain.values, label="Accuracy (train)")
            ax[num_ax].plot(self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].J.values, self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].AccuracyTest.values, label="Accuracy (test)")
            ax[num_ax].plot(self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].J.values, self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].RecallTest.values, label="Recall (test)")
            ax[num_ax].plot(self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].J.values, self.resultats_rainj_climat[self.resultats_rainj_climat.Climat==i].AUC.values, label="AUC")
            ax[num_ax].set_title(f"Variables de Performances pour climat {i} ({self.lib_climats[i]})")
            ax[num_ax].axhline(.9, color = '#666', linestyle = '--')
            ax[num_ax].axhline(.8, color = '#666', linestyle = '--')
            ax[num_ax].axhline(.7, color = '#666', linestyle = '--')
            ax[num_ax].axhline(.6, color = '#666', linestyle = '--')
            #ax[num_ax].axhline(.55, color = '#666', linestyle = '--')
            ax[num_ax].axhline(.5, color = '#666', linestyle = '--')
            ax[num_ax].set_xticks(xtick)
            ax[num_ax].legend(loc="upper right")

            num_ax+=1
        plt.show();

    # affiche les pvalue pour une liste de locations, selon un modele choisi
    def affiche_pvalue_rainj_locations(self, locations:list=[], data=None, modele_micro:bool=False, modele_climat:bool=False, climat:int=0):
        #self.resultats_rainj_location = pd.read_csv("resultats_rainj_location.csv")

        if data is None:
            if modele_micro:
                data = pd.read_csv("resultats_rainj_location.csv")
            if modele_climat:
                data = pd.read_csv("resultats_rainj_climat_location.csv")
        
        if len(locations)>0:
            liste = locations
        else:
            liste = self.data[self.data.Climat==climat].Location.unique()

        figure, ax = plt.subplots(len(liste), 1, figsize=(25, len(liste)*4))
        xtick = [*range(1,8), *range(8,365,7)]
        num_ax=0
        
        for i in np.sort(liste):
            
            ax[num_ax].plot(data[data.Location==i].J.values, data[data.Location==i].pvalue05.values, label="p-value d'après prédictions selon seuil par défaut (0,5)")
            ax[num_ax].plot(data[data.Location==i].J.values, data[data.Location==i].pvaluebest.values, label="p-value d'après prédictions selon seuil optimal")
            ax[num_ax].set_title(f"pvalues pour {i}")
            ax[num_ax].axhline(.05, color = 'r', linestyle = '--')
            ax[num_ax].set_xticks(xtick)
            ax[num_ax].legend(loc="upper right")
            ax[num_ax].set_ylim(-0.05,1.05)

            num_ax+=1
            
        if modele_climat:
            modele_lib = "Modélisation par zone climatique"
        if modele_micro:
            modele_lib = "Modélisation par location"
            
        if len(locations)==0:
            plt.suptitle(f"P-Value pour la zone de climat {climat} ({self.lib_climats[climat]})\n{modele_lib}", fontsize='xx-large')
        plt.show();
        
        
    # affiche les variables de performances pour une liste de locations, selon un modele choisi
    def affiche_perfs_rainj_locations(self, locations:list=[], data=None, modele_micro:bool=False, modele_climat:bool=False, climat:int=0):
        #self.resultats_rainj_location = pd.read_csv("resultats_rainj_location.csv")

        if data is None:
            if modele_micro:
                data = pd.read_csv("resultats_rainj_location.csv")
            if modele_climat:
                data = pd.read_csv("resultats_rainj_climat_location.csv")
        
        if len(locations)>0:
            liste = locations
        else:
            liste = self.data[self.data.Climat==climat].Location.unique()
        
        figure, ax = plt.subplots(len(liste), 1, figsize=(25, len(liste)*4))
        #figure, ax = plt.subplots(self.resultats_rainj_climat.Climat.nunique()+1, 1, figsize=(25,20))
        
        xtick = [*range(1,8), *range(8,365,7)]
        num_ax=0       
        
        for i in np.sort(liste):
        #for i in []:
            ax[num_ax].plot(data[data.Location==i].J.values, data[data.Location==i].AccuracyTrain.values, label="Accuracy (train)")
            ax[num_ax].plot(data[data.Location==i].J.values, data[data.Location==i].AccuracyTest.values, label="Accuracy (test)")
            ax[num_ax].plot(data[data.Location==i].J.values, data[data.Location==i].RecallTest.values, label="Recall (test)")
            #ax[num_ax].plot(data[data.Location==i].J.values, data[data.Location==i].AUC.values, label="AUC")
            ax[num_ax].set_title(f"Variables de Performances pour {i}")
            ax[num_ax].axhline(.9, color = '#aaa', linestyle = '--')
            ax[num_ax].axhline(.8, color = '#aaa', linestyle = '--')
            ax[num_ax].axhline(.7, color = '#aaa', linestyle = '--')
            ax[num_ax].axhline(.6, color = '#aaa', linestyle = '--')
            ax[num_ax].axhline(.5, color = '#555', linestyle = '--')
            ax[num_ax].set_xticks(xtick)
            ax[num_ax].legend(loc="upper right")
            ax[num_ax].set_ylim(-0.05,1.05)

            num_ax+=1

        if modele_climat:
            modele_lib = "Modélisation par zone climatique"
        if modele_micro:
            modele_lib = "Modélisation par location"

        if len(locations)==0:
            plt.suptitle(f"Variables de Performances pour la zone de climat {climat} ({self.lib_climats[climat]})\n{modele_lib}", fontsize='xx-large')
        plt.show();

    # ---
    # fais une prediction à partir d'un modele entrainé sur une ville pour une date donnée    
    # affiche variables explicatives de shap
    # ---
    def predit(self, date:str):
        X = self.X[self.y.index==date]
        y = self.y[self.y.index==date]

        # deja scalé lors de la preparation
        #if self.scaler!=None:
        #    X = self.scaler.transform(X)
            
        y_pred = self.modele.predict_proba(X)[:,1]>=self.res_roc_best_seuil

        dict_pluie={0:'Pas de pluie', 1:'Pluie'}
        
        s_exacte="fausse"
        if y.iloc[0] == y_pred[0]:
            s_exacte="exacte"

        if not hasattr(self, "resultats_pred"):
            self.resultats_pred = pd.DataFrame(columns=["Reel", "Predit", "Exact"])
        self.resultats_pred.loc[date] = [y.iloc[0], y_pred[0], (y_pred[0] == y.iloc[0])]

        print (f"Prédiction {s_exacte} - Jour: {date} Réel: {dict_pluie[y.iloc[0]]} Prédit: {dict_pluie[y_pred[0]]}")
        
        
        shap_values = self.shap_explainer(X)
        figure = plt.figure(figsize=(30,6))
        shap.plots.waterfall(shap_values[0], max_display=20, show=False)
        plt.tight_layout()
        plt.title(f"Prévision pour le {date} {s_exacte} - Réel: {dict_pluie[y.iloc[0]]} - Prédit: {dict_pluie[y_pred[0]]}")
        plt.show();
        
    
    def predit_mois(self):
        for i in range(1,31):
            self.predit(f"2010-09-{i:02d}")
        
    # -------
    #  verification de significative des predictions
    # -------   
        
    # fais un test de Chi2 pour verifier significativite des resultats
    def verification_significativite(self, y_reel, y_pred):
        from scipy.stats import chi2_contingency
        tcd = pd.crosstab(y_reel, y_pred)
        _,pval,_,_ =chi2_contingency(tcd)
        
        return pval
        
    # retourne le tuple des pvalue du test de Chi2 pour les predictions selon le seuil par défaut et celui optimal
    def verification_significativite_modele(self):
        y_pred_proba = self.modele.predict_proba(self.X_test)
        
        pval05   = self.verification_significativite(self.y_test, y_pred_proba[:,1] >= .5)
        pvalbest = self.verification_significativite(self.y_test, y_pred_proba[:,1] >= self.res_roc_best_seuil)
        
        return (pval05, pvalbest)
     
    # affiche matrice de prediction d'un modele
    def _modelisation_matrice_confusion(self, modele):
        y_pred = modele.predict(self.X_test)
        print(pd.crosstab(self.y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
        
    # verifie la significative de toutes les locations
    def verifice_significativite_micro(self):
        couple_loc_cli = self.data[["Location", "Climat"]].drop_duplicates()
        couple_loc_cli_triee = couple_loc_cli.sort_values(["Climat", "Location"])
        couple_loc_cli_triee.apply(lambda row:self.predictions2016(climat=row.Climat, location=row.Location, modele_micro=True), axis=1)
        #self.verification_significativite(self.y_test, self.y_test)        
       
    # affiche metriques de classification
    def scores_classification(self, y_reel, y_pred):
        print (f"\nMétriques de clasification :\nAccuracy: {accuracy_score(y_reel, y_pred):.4f} - Recall: {recall_score(y_reel, y_pred):.4f} - Precision: {precision_score(y_reel, y_pred):.4f} - Score F1: {f1_score(y_reel, y_pred):.4f}\n")
        print ("Matrice de confusion :")
        print(pd.crosstab(y_reel, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
        print ("---------------------------------")

       
    # --------------------------------
    # --------------------------------
    # --------------------------------
    #  modelisation: fonction la plus importante du module
    # --------------------------------
    # --------------------------------
    # --------------------------------
    
    def modelisation(self, nom_modele:str="XGBoost", cible:str="RainTomorrow", gs:bool=True, climat:int=None, location:str="", totalite:bool=True, cut2016:bool=False):
               
        # seules les variables débutant par Rain impliquent de la classification, sauf Rainfall
        est_classification = cible.startswith("Rain") and not cible.startswith("Rainfall")
                
        self.cible = cible
        
        print (time.ctime())
        
        param_modele=None
        existe_proba=False
        modele=None
        hp=""

        print (f'\n -------\nModelisation de {cible} avec un {nom_modele}\n -------\n')
        i_temps_debut=time.time()
        
        self._modelisation_preparation(cible, True, climat, location, cut2016=cut2016)
        
        
        if nom_modele=='KNeighborsClassifier':
                        
            existe_proba=True
            modele = KNeighborsClassifier(n_neighbors=1)
            param_modele={ 'n_neighbors': np.arange(1,6)
                      }          
            
        elif nom_modele=='DecisionTreeClassifier':
            modele = DecisionTreeClassifier(random_state=0, max_depth=150)      

        elif nom_modele=='RandomForestClassifier':
        
            existe_proba=True
            modele  = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=1)            
            param_modele={ 'max_depth': range(10,21,5),
                          #'n_estimators': [5, 10, 50, 400, 800]
                          'n_estimators': [50, 100, 150]
                          }
            
        
        # boosting (chaque machine apprend à ajuster les erreurs de la precedente)
        elif nom_modele=='GradientBoostingClassifier':
            existe_proba=True
            modele=GradientBoostingClassifier(random_state=0, n_estimators=20, max_depth=4) 
            param_modele={ 'max_depth': range(9,30,1),
                          #'n_estimators': [5, 10, 50, 400, 800]
                          'n_estimators': [5, 10]
                          }

                # xgboost
        elif nom_modele=='XGBoost':
            if est_classification:
                existe_proba=True
                modele=xgb.XGBClassifier(random_state=0, learning_rate=.23, n_estimators=50, max_depth=6) 

                # modele global:
                param_modele={ 'learning_rate': [.23],
                              'n_estimators': [ 50],#, 50],
                              'max_depth': [ 6],#, 7, 8, 9, 10],
                              'random_state':[0]
                              }

                """
                # best Quyen
                param_modele={'max_depth': [9],
                 'min_child_weight': [8.266409668869114],
                 'subsample': [0.874748527185061],
                 'reg_alpha': [6.968394128293252],
                 'reg_lambda': [1.672170373377269],
                 'gamma': [0.0073317410705931495],
                 'learning_rate': [0.10186418984471239]}
                """

                # modele micro
                if location !="":
                    param_modele={ 'learning_rate': [.2],
                                  'n_estimators': [ 10],#, 50],
                                  'max_depth': [ 3],#, 7, 8, 9, 10],
                                  'random_state':[0]
                                  }
                
                # modele climat
                if climat!=None:
                    param_modele={ 'learning_rate': [.2],
                                  'n_estimators': [ 30],#, 50],
                                  'max_depth': [ 4],#, 7, 8, 9, 10],
                                  'random_state':[0]
                                  }               
                
            else:
                modele=xgb.XGBRegressor(random_state=0, learning_rate=.1, n_estimators=100, max_depth=4)

        elif nom_modele=='MLPClassifier':
            
            # max 1000, hidden 1200, alpha 0.0001 : 70 71 79 90 85 (170 mn)
            existe_proba=True
            # sgd = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(1200,), alpha=0.0001) # 74%, 8mn / 97%             
            modele = MLPClassifier(random_state=5, max_iter=300, hidden_layer_sizes=(100,100), verbose=0) # 74%, 8mn / 97%
            param_modele = {'hidden_layer_sizes': [(50,200,50), (100,100), (100,)],
                            #'activation': ['tanh', 'relu'],
                            #'solver': ['sgd', 'adam']
                            
                            }
        
        elif nom_modele=='DummyClassifier':
            existe_proba=True
            modele=DummyClassifier(random_state=0, strategy='stratified')
            
        elif nom_modele=='LogisticRegression':
            existe_proba=True
            modele=LogisticRegression(random_state=0)
            
        else:
            print("\n -------\nNom de modèle inconnu\n -------\n")
            return
        
        # gridsv
        if param_modele!=None and gs:
            outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
            resc = make_scorer(recall_score,pos_label=1) # la difficulte est de predire correctemetnt les jours où il pleut reellement => il faut optimiser le recall sur la cible 1
            
            gcv = GridSearchCV(estimator=modele, 
                        param_grid=param_modele,
                        #scoring='recall',
                        #scoring = resc,
                        #scoring=score_callback,
                        scoring='roc_auc',
                        verbose=0,
                        cv=outer_cv,
                        n_jobs=-1
                        )
    
            gcv.fit(self.X_train, self.y_train)
            self.gcv = gcv

            """
            for params in tqdm(param_modele):
                i_debut_fit=time.time()
                gcv.set_params(**params)
                gcv.fit(self.X_train, self.y_train)
                i_fin_fit=time.time()
                print(f"Hyper paramètres: {params} - score:{gcv.cv_results_['mean_test_score'][-1]}\n")
                print(f"Temps fit: {(i_fin_fit-i_debut_fit)/60:.2f} minutes\n\n")
            """  
            
            modele = gcv.best_estimator_
            print (gcv.best_params_)
            hp = gcv.best_params_
            self.gcv = gcv
    
        else:
            modele.fit(self.X_train, self.y_train)
        
        self.modele=modele
        self.titre_modele = self.titre_graphe(nom_modele, hp, climat, location, cible)

        # n'execute la suite que si on veut le traitement total
        # pour des traitements successifs, on s'arrete là
        if not totalite:
            return

        
        print('Modele ', type(modele))
        
        predictions=modele.predict(self.X_train)
        i_fin_train=time.time()

        predictions=modele.predict(self.X_test)
        i_fin_test=time.time()

        # s'il s'agit de classfication:        
        if est_classification:
        
            print ("Accuracy train: {:.2f}% - Temps train: {:.2f} minutes".format(modele.score(self.X_train, self.y_train)*100, (i_fin_train-i_temps_debut)/60))
            print ("Accuracy test: {:.2f}% - Temps test: {:.2f} minutes".format(modele.score(self.X_test, self.y_test)*100, (i_fin_test-i_fin_train)/60))
    #        print (f"\nScore F1: {f1_score(self.y_test, predictions):.2f} - Accuracy: {accuracy_score(self.y_test, predictions):.2f} - Recall: {recall_score(self.y_test, predictions):.2f} - Precision: {precision_score(self.y_test, predictions):.2f}\n")
            
            
            self._modelisation_matrice_confusion(self.modele)
        
            predictions_proba=np.zeros(shape=predictions.shape)
            if existe_proba:
                predictions_proba=modele.predict_proba(self.X_test)
              
            # renvoie les predictions sur le jeu complet
            #predictions=modele.predict(t_donnees_completes)
            #predictions_proba=np.zeros(shape=predictions.shape)
    #        if existe_proba:
    #╩            predictions_proba=modele.predict_proba(t_donnees_completes).max(axis=1)
            self.pp = predictions_proba        
    
            self.trace_courbe_roc(modele, self.titre_modele)
            
            # test le caractere significatif des resultats
            print ("P-value pour Test de X² avec seuil par défaut de 0,5: ", self.verification_significativite(self.y_test, predictions))
            print (f"Test de X² avec seuil optimal ({self.res_roc_best_seuil:.2f}): ", self.verification_significativite(self.y_test, predictions_proba[:,1] >= self.res_roc_best_seuil))

            # scores            
            print ("Seuil par défaut:")
            self.scores_classification(self.y_test, predictions_proba[:,1]>=.5)
            print ("\nSeuil optimal:")
            self.scores_classification(self.y_test, predictions_proba[:,1]>=self.res_roc_best_seuil)
            

        # s'il s'agit de regression
        else:
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            mae = mean_absolute_error(self.y_test, predictions)
            print (f"\n ----- \n RMSE : {rmse:.2f} - MAE : {mae:.2f}\n ----- \n\n")

        i_fin_completes=time.time()
        print (" Temps comp: {:.2f} minutes".format( (i_fin_completes-i_fin_test)/60))
                
        print (time.ctime())
      
    # modelise un DNN
    def modelisation_dnn(self, nom_modele:str="DNN", cible:str="RainTomorrow", gs:bool=True, climat:int=None, location:str="", totalite:bool=True, cut2016:bool=False):
        # seules les variables débutant par Rain impliquent de la classification, sauf Rainfall
        est_classification = cible.startswith("Rain") and not cible.startswith("Rainfall")
                
        print (time.ctime())      

        print (f'\n -------\nModelisation de {cible} avec un {nom_modele}\n -------\n')
        i_temps_debut=time.time()
        
        self._modelisation_preparation(cible, True, climat, location, cut2016=cut2016)

        # fait egalement un MinMax pour que les valeurs soient entre 0 et 1
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.X)
        
        self.X_train=minmax_scaler.transform(self.X_train)
        self.X_test=minmax_scaler.transform(self.X_test)    

        inputs = Input(shape=(self.X_train.shape[1]), name="Inputlay")
        dense1 = Dense(units=50, activation='tanh', name='d1')
        dense2 = Dense(units=50, activation='relu', name='d2')

#        dense1 = Dense(units=64, activation='relu', name='d1')
#        dense2 = Dense(units=64, activation='relu', name='d2')
        #dense3 = Dropout(.2)

#        dense3 = Dense(units=50, activation='relu', name='d3')
        dense4 = Dense(units=1, activation='sigmoid', name='d4')
        
        x = dense1(inputs)
        x = dense2(x)
        #x = dense3(x)

#        x= inputs
        outputs = dense4(x)
        
        modele = Model(inputs = inputs, outputs = outputs)
        modele.summary()
        
        opt = Adam(lr=1e-4)
#        opt = Adam(lr=1e-1)
        
        modele.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=opt)

        # gere le desequilibre        
        from sklearn.utils import compute_class_weight
        classWeight = compute_class_weight('balanced', classes=[0,1], y=self.y_train) 
        classWeight = dict(enumerate(classWeight))
        self.classWeight = classWeight
        
        classWeight = None
        #classWeight = {0:100, 1:1}
        
        lrate = LearningRateScheduler(self.learning_rate_schedule)
        #cb_liste=[lrate, tf.keras.callbacks.EarlyStopping(patience=50)]
        cb_liste=[lrate]
        #cb_liste = None
        
        #history = modele.fit(self.X_train, self.y_train, epochs=1000, batch_size=128, validation_split=.2, class_weight=classWeight, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=50)])
        history = modele.fit(self.X_train, self.y_train, epochs=1000, batch_size=128, validation_split=.2, class_weight=classWeight, verbose=2, callbacks=cb_liste)
        
#        history = modele.fit(self.X_train, self.y_train, epochs=50, batch_size=50, validation_split=.2, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
        
        self.modele=modele
        self.history=history
        
        i_fin_completes=time.time()
        print (" Temps comp: {:.2f} minutes".format( (i_fin_completes-i_temps_debut)/60))
        
        print (time.ctime())
        self.resultats_dnn(climat, location, cible)
        
    # learning rate schedule
    def learning_rate_schedule(self, epoch, lr):
        lrate = .1
        if epoch<30:
            lrate=1e-3
        elif epoch<200:
            lrate=1e-4
#        elif epoch<250:
#            lrate=1e-5
#        else:
#            lrate=1e-6
        else:
            lrate=1e-5
    
        
        #print ("LR: ",lrate)
        return lrate
    
        
    # affiche les resultats du DNN entrainé
    def resultats_dnn(self, climat:int=None, location:str="", cible:str="RainTomorrow"):
        
        nom_modele = self.titre_graphe("DNN", "", climat, location, cible)
        
        plt.figure(figsize=(16, 6))
        plt.plot(self.history.history['val_binary_accuracy'], "g", label="Accuracy (Val)")
        plt.plot(self.history.history['binary_accuracy'], "b", label="Accuracy (Train)")
        plt.ylim((.85,.88))
        plt.axhline(y=0.78, color='gray', linestyle='dashed')        
        plt.axhline(y=0.79, color='gray', linestyle='dashed')        
        plt.axhline(y=0.8, color='black', linestyle='dashed')        
        plt.axhline(y=0.81, color='gray', linestyle='dashed')        
        plt.axhline(y=0.82, color='gray', linestyle='dashed')        
        plt.axhline(y=0.83, color='gray', linestyle='dashed')        
        plt.axhline(y=0.835, color='#CCC', linestyle='dashed')        
        plt.axhline(y=0.84, color='gray', linestyle='dashed')        
        plt.axhline(y=0.845, color='#CCC', linestyle='dashed')        
        plt.axhline(y=0.85, color='black', linestyle='dashed')        
        plt.axhline(y=0.855, color='#CCC', linestyle='dashed')        
        plt.axhline(y=0.86, color='gray', linestyle='dashed')        
        plt.axhline(y=0.865, color='#CCC', linestyle='dashed')        
        plt.axhline(y=0.87, color='gray', linestyle='dashed')        
        plt.axhline(y=0.875, color='#CCC', linestyle='dashed')        
        plt.axhline(y=0.88, color='gray', linestyle='dashed')        
        plt.axhline(y=0.89, color='gray', linestyle='dashed')        
        plt.axhline(y=0.9, color='black', linestyle='dashed')        

        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Historique d'Accuracy \n{nom_modele}")
        plt.show();
        
        self.test_pred = self.modele.predict(self.X_test)
        self.test_pred_class = self.test_pred>.5
        
        print(classification_report(self.y_test, self.test_pred_class))
        print (confusion_matrix(self.y_test, self.test_pred_class))

        print(self.modele.evaluate(self.X_test, self.y_test))
        
        self.trace_courbe_roc_ann(self.y_test, self.test_pred, nom_modele)
        
        print ("Seuil par défaut:")
        self.scores_classification(self.y_test, self.test_pred.reshape(-1)>=.5)
        print ("\nSeuil optimal:")
        self.scores_classification(self.y_test, self.test_pred.reshape(-1)>=self.res_roc_best_seuil)
        
        # loss
        plt.figure(figsize=(16, 6))
        plt.plot(self.history.history['val_loss'], "g", label="Loss (Val)")
        plt.plot(self.history.history['loss'], "b", label="Loss (Train)")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Evolution de la fonction de perte \n{nom_modele}")
        plt.show();
        
        
        print (time.ctime())

    # --------
    # courbes ROC
    # --------

    # trace la courbe roc d'un DNN
    def trace_courbe_roc_ann(self, y_reel, y_pred_proba, titre_graphe:str=""):
        fpr, tpr, thresholds = roc_curve(y_reel, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # cherche seuil optimal
        diff_pr = tpr - fpr
        best_i_seuil = np.argmax(diff_pr)
        best_seuil = thresholds[best_i_seuil]
        self.res_roc_best_seuil = best_seuil
        
        # cherche valeurs pour 0,5
        i_05 = np.abs(thresholds-.5).argmin()
        fpr_05=fpr[i_05]
        tpr_05=tpr[i_05]     
        
        plt.figure(figsize=(12, 9))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[best_i_seuil], tpr[best_i_seuil], 'ro', markersize=8, label=f'Seuil Optimal = {best_seuil:.2f}')
        plt.plot(fpr[i_05], tpr[i_05], 'ko', markersize=8, label=f'Seuil par défaut = 0.5')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux Faux Positifs')
        plt.ylabel('Taux Vrais Positifs')
        
            
        plt.title(f'Courbe ROC \n{titre_graphe}')
        plt.legend(loc='lower right')
        plt.show()
        
    # trace la courbe roc d'un modele standard        
    def trace_courbe_roc(self, modele, titre_graphe:str=""):
        y_test_pred_proba = modele.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_test_pred_proba[:,1])
        roc_auc = auc(fpr, tpr)
        self.res_roc_auc = roc_auc
        
        # cherche seuil optimal
        diff_pr = tpr - fpr
        best_i_seuil = np.argmax(diff_pr)
        best_seuil = thresholds[best_i_seuil]
        self.res_roc_best_seuil = best_seuil
        
        # cherche valeurs pour 0,5
        i_05 = np.abs(thresholds-.5).argmin()
        fpr_05=fpr[i_05]
        tpr_05=tpr[i_05]     
        
        # commenter ou afficher selon si je veux ou non appeler entrainer massivement des modeles
        plt.figure(figsize=(12, 9))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[best_i_seuil], tpr[best_i_seuil], 'ro', markersize=8, label=f'Seuil Optimal = {best_seuil:.2f}')
        plt.plot(fpr[i_05], tpr[i_05], 'ko', markersize=8, label=f'Seuil par défaut = 0.5')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux Faux Positifs')
        plt.ylabel('Taux Vrais Positifs')
        
            
        plt.title(f'Courbe ROC \n{titre_graphe}')
        plt.legend(loc='lower right')
        plt.show()
        
        y_pred_seuil = modele.predict_proba(self.X_test)[:,1] >= best_seuil       
        y_pred = modele.predict(self.X_test)
        
        self.res_acc_test_seuil = accuracy_score(self.y_test, y_pred_seuil)
        self.res_recall_test_seuil = recall_score(self.y_test, y_pred_seuil)
        self.res_precision_test_seuil = precision_score(self.y_test, y_pred_seuil)
        self.res_f1_test_seuil = f1_score(self.y_test, y_pred_seuil)

        self.res_acc_test = accuracy_score(self.y_test, y_pred)
        self.res_recall_test = recall_score(self.y_test, y_pred)
        self.res_precision_test = precision_score(self.y_test, y_pred)
        self.res_f1_test = f1_score(self.y_test, y_pred)

        # calcul aussi sur le train pour s'assurer qu'il n'y ait pas d'overfit
        y_train_pred_seuil = modele.predict_proba(self.X_train)[:,1] >= best_seuil       
        self.res_acc_train_seuil = accuracy_score(self.y_train, y_train_pred_seuil)
        y_train_pred = modele.predict(self.X_train)
        self.res_acc_train = accuracy_score(self.y_train, y_train_pred)
        
        print (f"\nAvec seuil par défaut:\nScore F1: {self.res_f1_test:.4f} - Accuracy: {self.res_acc_test:.4f} - Recall: {self.res_recall_test:.4f} - Precision: {self.res_precision_test:.4f}\n")
        
        print (f"\nAvec seuil optimisé:\nScore F1: {self.res_f1_test_seuil:.4f} - Accuracy: {self.res_acc_test_seuil:.4f} - Recall: {self.res_recall_test_seuil:.4f} - Precision: {self.res_precision_test_seuil:.4f}\n")
        print (f"Matrice de confusion avec seuil de {best_seuil:.2f}:")
        print(pd.crosstab(self.y_test, y_pred_seuil, rownames=['Classe réelle'], colnames=['Classe prédite']))

        print(classification_report_imbalanced(self.y_test, y_pred_seuil))

    # deduit le titre d'un graphe selon les pricipales variables
    def titre_graphe(self, nom_modele:str, hp:str, climat:int=None, location:str="", cible:str=""):
        titre_clim_loc=""        
        if climat is not None:
            titre_clim_loc="\nClimat: "+str(climat)+" ("+self.lib_climats[climat]+")"
        if location!="":
            titre_clim_loc="\nLocation: "+location
            
        return f'Modèle {nom_modele} \n {hp}{titre_clim_loc} \n Variable cible:{cible}'

    def AUC_nb_J(self, nom_modele:str="XGBoost", cible:str="RainTomorrow", gs:bool=False, climat:int=None, location:str="", nbj:int=8):
        scores_auc=[]
        
        for j in range(1,nbj):
            v_cible = f"Rain_J_{j:02d}"
            self.modelisation(nom_modele, v_cible, gs, climat, location, totalite=False )

            # calcule AUC
            y_test_pred_proba = self.modele.predict_proba(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_test_pred_proba[:,1])
            roc_auc = auc(fpr, tpr)
            scores_auc.append(roc_auc)
            
        #self.scores_auc=scores_auc
        return scores_auc        


    def AUC_macro(self, nom_modele:str="XGBoost", cible:str="Rain", gs:bool=False, nbj:int=8):
        all_scores_auc=self.AUC_nb_J(nom_modele, cible, gs=gs, nbj=nbj)
            
        self.AUC_trace(all_scores_auc, mode="macro", nbj=nbj)

    def AUC_par_climat(self, nom_modele:str="XGBoost", cible:str="Rain", gs:bool=False, nbj:int=8):
        all_scores_auc=[]
        climats=[]
        for climat in np.sort(self.data.Climat.unique()):
            all_scores_auc.append(self.AUC_nb_J(nom_modele, cible, gs=gs, climat=climat, nbj=nbj))
            climats.append(climat)
            
        self.AUC_trace(all_scores_auc, climats, nbj=nbj)

    def AUC_par_location(self, nom_modele:str="XGBoost", cible:str="Rain", gs:bool=False, climat:int="", nbj:int=8):
        all_scores_auc=[]
        locations=[]       
        
        # si un climat est defini, on affichera les villes de ce climat uniquement
        data = self.data
        if climat!="":
            liste_locations=self.data[self.data.Climat==climat].Location.unique()
        else:
            liste_locations=self.data.Location.unique()
        
        for location in liste_locations:
            all_scores_auc.append(self.AUC_nb_J(nom_modele, cible, gs=gs, location=location, nbj=nbj))
            locations.append(location)
            
        self.AUC_trace(all_scores_auc, locations, mode="Location", nbj=nbj)

        
    def AUC_trace(self, scores_auc, types=None, mode:str="Climat", nbj:int=8):
        fig = plt.figure(figsize=(12,8))
        
        if mode=="Climat":
            for score_auc, item_type in zip(scores_auc, types):
                plt.plot(range(1,nbj), score_auc, label=f"Climat {item_type} ({self.lib_climats[item_type]})", color=self.palette[item_type])
        elif mode=="Location":
            for score_auc, item_type in zip(scores_auc, types):
                plt.plot(range(1,nbj), score_auc, label=f"{item_type}")
        else:
            plt.plot(range(1,nbj), scores_auc, label="Global")
            
        plt.ylabel("AUC")
        plt.xlabel("Numéro de la journée à J+n prédite pour RainToday")
        plt.title("Score AUC en fonction du décalage de prédiction de pluie dans le futur\n")
        plt.legend(loc='upper right')
        plt.ylim(.45,1)
        plt.axhline(y=0.5, color='gray', linestyle='dashed')
        plt.show();
        
    # regressions
    
    def RMSE_nb_J(self, nom_modele:str="XGBoost", cible:str="MaxTemp", gs:bool=False, climat:int=None, location:str="", nbj:int=8):
        scores_rmse=[]
        scores_mae=[]
        
        for j in range(1,nbj):
            v_cible = f"{cible}_J_{j:02d}"
            self.modelisation(nom_modele, v_cible, gs, climat, location, totalite=False )

            predictions=self.modele.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            mae = mean_absolute_error(self.y_test, predictions)

            scores_rmse.append(rmse)
            scores_mae.append(mae)
            
        return scores_rmse, scores_mae        

    
    def RMSE_par_climat(self, nom_modele:str="XGBoost", cible:str="MaxTemp", gs:bool=False, nbj:int=8):
        all_scores_rmse=[]
        all_scores_mae=[]
        climats=[]
        for climat in self.data.Climat.unique():
            rmse, mae = self.RMSE_nb_J(nom_modele, cible, gs=gs, climat=climat, nbj=nbj)
            all_scores_rmse.append(rmse)
            all_scores_mae.append(mae)
            
            climats.append(climat)
            
        self.scores_trace(all_scores_rmse, climats, nbj=nbj, cible=cible, libelle="RMSE")
        self.scores_trace(all_scores_mae, climats, nbj=nbj, cible=cible, libelle="MAE")
        
    def RMSE_par_location(self, nom_modele:str="XGBoost", cible:str="MaxTemp", gs:bool=False, climat:int="", nbj:int=8):
        all_scores_rmse=[]
        all_scores_mae=[]
        locations=[]       
        
        # si un climat est defini, on affichera les villes de ce climat uniquement
        data = self.data
        if climat!="":
            liste_locations=self.data[self.data.Climat==climat].Location.unique()
        else:
            liste_locations=self.data.Location.unique()
        
        for location in liste_locations:
            rmse, mae = self.RMSE_nb_J(nom_modele, cible, gs=gs, location=location, nbj=nbj)
            all_scores_rmse.append(rmse)
            all_scores_mae.append(mae)
            
            locations.append(location)
            
        self.scores_trace(all_scores_rmse, locations, mode="Location", nbj=nbj, cible=cible, libelle="RMSE")
        self.scores_trace(all_scores_mae, locations, mode="Location", nbj=nbj, cible=cible, libelle="MAE")

        
        
    def scores_trace(self, scores, types, mode:str="Climat", nbj:int=8, cible:str="MaxTemp", libelle="RMSE"):
        fig = plt.figure(figsize=(12,8))

        if mode=="Climat":
            for score, item_type in zip(scores, types):
                plt.plot(range(1,nbj), score, label=f"Climat {item_type} ({self.lib_climats[item_type]})", color=self.palette[item_type])
        else:
            for score, item_type in zip(scores, types):
                plt.plot(range(1,nbj), score, label=f"{item_type}")
            
        plt.ylabel(libelle)
        plt.xlabel(f"Numéro de la journée à J+n prédite pour {cible}")
        plt.title(f"{libelle} en fonction du décalage de prédiction dans le futur\n")
        plt.legend(loc='upper right')
        #plt.ylim(.45,1)
        #plt.axhline(y=0.5, color='gray', linestyle='dashed')
        plt.show();              
        
        
    def trace_shap(self, nom_modele:str="XGBoost"):
        # =============================================================================================
        # SHAP - waterfall
        # =============================================================================================
        
        if nom_modele=="XGBoost" or nom_modele=="RandomForestClassfifier":
            shap_explainer = shap.TreeExplainer(self.modele, feature_names=self.X.columns)
        if nom_modele=="DNN":
            shap_explainer = shap.Explainer(self.modele, self.X_train, feature_names=self.X.columns)

        self.shap_explainer = shap_explainer
        
        shap_values = shap_explainer(self.X_test)
        #shap_df = pd.DataFrame(shap_values.values, columns=self.X.columns)
               
        s = 0
        
        figure, ax = plt.subplots(1, 5, figsize=(30,6))
        ax = ax.reshape(-1)
        i=1

        for s in range(5):
            plt.figure()
            #[plt.subplot(1,5,i)
            shap.plots.waterfall(shap_values[s], max_display=20, show=False)
            #plt.tight_layout()
            i+=1
        plt.show();
        
#            file_out = f'{path_output}/waterfall_sample_{s}.png'
#            plt.savefig(file_out)
        
        # =============================================================================================
        # SHAP - Mean SHAP Plot
        # =============================================================================================
        
        plt.figure()
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.tight_layout()
#        file_out = f'{path_output}/mean_SHAP_barplot.png'
#        plt.savefig(file_out)
        
        # =============================================================================================
        # SHAP - Beeswarm Plot
        # =============================================================================================
        
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()
#        file_out = f'{path_output}/beeswarm.png'
#        plt.savefig(file_out)
        plt.show();
        # =============================================================================================
        # SHAP - Dependence Plots
        # =============================================================================================
        
        list_variables = ["Humidity3pm", "Pressure3pm", "WindGustSpeed", "Sunshine", "RainToday", "Rainfall", "MaxTemp", 
                          "AmplitudeTemp", "SaisonCos4pi", "SaisonCos2pi", "Climat_0", "Climat_1", "Climat_2", "Climat_3", "Climat_4", "Climat_6",
                          "WindGustDir_X", "WindGustDir_Y"
                          ]
#        for variable in list_variables:
        self.shap_values=shap_values
    
        n_cote = int(np.sqrt(len(self.X.columns)))+1
        figure, ax = plt.subplots(n_cote, n_cote, figsize=(30,30))
        ax = ax.reshape(-1)
        i=1
        for variable in self.X.columns:
            #plt.figure()
            
            shap.plots.scatter(shap_values[:, variable], show=False, ax=ax[i])
            
            #plt.tight_layout()
            #ax[i].tight_layout()
            i+=1
#            file_out = f'{path_output}/dependence_{variable}.png'
#            plt.savefig(file_out)
        plt.show();
        
    # ---- series temporelles
    
    def prepare_serie_temporelle(self, location:str="", variable:str="MaxTemp", affiche=True):
        df = self.data
        if location!="":
            df = self.data.loc[self.data.Location==location]

        # on ne reprend pas plus tôt à cause des trous sur 3 mois
        df = df.loc[df.index>='2013-03-03']
        #df = df.loc[df.index>='2009-01-01']

        self.serie_temporelle=df[variable]
        self.titre_analyse = location+str(" - ")+variable
        
        if affiche:
            plt.figure(figsize=(16,8))
            plt.plot(self.serie_temporelle)
            plt.title(self.titre_analyse)
        
    def decompose_serie_temporelle(self):
        result = seasonal_decompose(self.serie_temporelle, model='additive', period=365)
        result.plot();

    def affiche_acf_pacf(self):                                 # darwin - mildura
        plot_acf(self.serie_temporelle.diff(1).dropna(), lags = 30)     # 3 - 4
        plot_acf(self.serie_temporelle.diff(365).dropna(), lags = 30)   # 10- 3

        plot_pacf(self.serie_temporelle.diff(1).dropna(), lags = 30)    # 5 - 12
        plot_pacf(self.serie_temporelle.diff(365).dropna(), lags = 30)  # 2 - 2
                       
    
    def applique_sarima(self, p=2, d=1, q=0, P=0, D=1, Q=1, ax=None):
        
        # tronconne la série
        serie_temporelle_debut = self.serie_temporelle.iloc[:-365]
        #serie_temporelle_fin = self.serie_temporelle.iloc[-12:]
        
        # SARIMAX
        print (time.ctime())

        smx = sm.tsa.SARIMAX(serie_temporelle_debut, order=(p,d,q), seasonal_order=(P,D,Q,365))
        smx_fitted = smx.fit()#maxiter=1000)

        print (time.ctime())

        print(smx_fitted.summary())
        self.smx = smx_fitted
        
#        pred = smx_fitted.predict(49, 61)
#        serie_predite = pd.concat([serie_temporelle_debut, pred])

#        self.sp = serie_predite

#        plt.figure(figsize=(16,8))
#        plt.plot(serie_predite)
#        plt.axvline(x=datetime.date(2020,12,15), color='orange')

        # avec intervalle de confiance
        
        prediction = smx_fitted.get_forecast(steps =365).summary_frame()  #Prédiction avec intervalle de confiance
        
        if ax==None:
            fig, ax = plt.subplots(figsize = (40,5))
            
        ax.plot(self.serie_temporelle)
        prediction['mean'].plot(ax = ax, style = 'k--') #Visualisation de la moyenne
        ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1); #Visualisation de l'intervalle de confiance    
        
        # affiche N-1
        last_12_months = self.serie_temporelle.shift(365)
        ax.plot(last_12_months[-365:], label="N-1")      
        
        ax.set_title(self.titre_analyse)
                
    # -----    
    # a deplacer en dataviz
    def animation_variable(self, variable:str="RainToday", discrete:bool=False):
        
        data = self.data.loc[(self.data.index>='2014-04-01')&(self.data.index<='2014-04-30'),:].copy()
        data["Date"] = data.index
        
        if discrete:
            cible = data[variable].astype(str)
        else:
            cible = data[variable]
        
        fig = px.scatter_mapbox(data, 
                                lat='lat', 
                                lon='lng', 
                                hover_name='Location', 
                                color=cible, 
                                #text='Location', 
                                #labels=modele.labels_, 
                                animation_frame="Date",
#                                animation_group="Location",
                                size_max=30, 
                                opacity=.8,
                                #color_continuous_scale=px.colors.qualitative.Plotly
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                #color_discrete_sequence=px.colors.qualitative.T10,
                                range_color=[data[variable].min(), data[variable].max()]
                                ).update_traces(marker=dict(size=30))
                
        fig.update_layout(mapbox_style='open-street-map')
        fig.show(renderer='browser')      

#source = pd.read_csv("data_process4_knnim_resample_J365.csv", index_col=0)
#source = pd.read_csv("data_process3_knnim_resample_J2.csv", index_col=0)

source = pd.read_csv("data_process5_knnim_resample_J2.csv", index_col=0)
#source = pd.read_csv("data_basique_location.csv", index_col=0)

#pm = ProjetAustralieModelisation(pd.read_csv("data_basique.csv", index_col=0))
pm = ProjetAustralieModelisation(source)

#pm.modelisation_dnn()

#pm.animation_variable()

# data_process_non_knnim => preprocession avancée (mais sans knni ni reequilibrage des classes)
# data_process4_knnim_resample_J365 => idem, mais 365j de prevision pour Rainfall, MaxTemp, RainToday. Knni SANS drop RainTomorrow (car plein de variables cibles possibles), SaisonCos
# data_process3_knnim_resample_J2 => version light
# data_process5_knnim_resample_J2 => dataset jusqu'au 25/6/17, ajout de SaisonCos2pi. Pas d'ajout des Rain_J

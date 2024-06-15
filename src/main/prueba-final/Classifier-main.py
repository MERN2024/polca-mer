# Clasificador principal

# ==============================================================================
import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import shutil
from sklearn.discriminant_analysis import StandardScaler
import Feature_Extractor_final as ext
import joblib

# ==============================================================================
ruta_archivo = input("Ingresa la ruta del archivo MP3: ")

print(f"El archivo se ha guardado en {ruta_archivo}")
ext.extract_feature(ruta_archivo)

data = pd.read_csv('temp.csv')
feature = data.loc[:, 'tempo':]
print('##### feature ######')
print(feature)

featureName = list(feature)
features = feature.values

print('##### features ######')
print(feature.values)

array = np.array(features)
print('##### array ######')
print(array)

scaler = StandardScaler()
X = array

print('##### X train ######')
print(X)

modelo = joblib.load('modelo_3.pkl')
predicciones = modelo.predict(X)
print(predicciones)

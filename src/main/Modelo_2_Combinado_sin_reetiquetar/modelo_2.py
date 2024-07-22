# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import pickle



# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('fivethirtyeight')

# Modelado
# ==============================================================================
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import multiprocessing

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Datos simulados
# ==============================================================================

data = pd.read_csv('src\main\Modelo_2_Combinado_sin_reetiquetar\modelo_2_descriptores.csv')
feature = data.loc[:, 'tempo':]
featureName = list(feature)
color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

for name in featureName:
    feature[name] = np.nan_to_num((feature[name]-feature[name].min())/(feature[name].max()-feature[name].min()),0)

features = feature.values

array = np.array(features)
print(array)
labels = data.loc[:, 'emocion'].dropna()

X = array
y = labels

test_size = 0.3
random_seed = 7

X_train, X_test, y_train, y_test  = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

print("Partición de entrenamento")
print("-----------------------")
print(y_train)
print(X_train)
print(X_train)
print(" ")

print("Partición de test")
print("-----------------------")
print(y_test)
print(X_test)
print(X_test)

# Modelos
# ==============================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo_2 = MLPClassifier(
                hidden_layer_sizes=( 300, 25, 5),
                learning_rate_init=0.01,
                solver = 'sgd',
                max_iter = 1000,
                random_state = 123
            )


modelo_2.fit(X=X_train, y=y_train)
file_route_3 = 'modelo_2.pkl'
with open(file_route_3, 'wb') as model_file:
    pickle.dump(modelo_2, model_file)
print('Terminó el entrenamiento')


# Accuracy
# ==============================================================================
result = []
predicciones = modelo_2.predict(X = X_test)
result.append(accuracy_score(predicciones, y_test)*100)
print('Accuracy_score: ', result)

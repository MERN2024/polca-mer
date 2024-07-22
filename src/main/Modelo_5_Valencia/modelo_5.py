# Prueba con atributos dimensionales (valencia)
#
# ==============================================================================
import numpy as np
import pandas as pd
import pickle
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
data_source = pd.read_csv('src\main\Modelo_5_Valencia\modelo_5_descriptores.csv')
feature = data_source.loc[:, 'tempo':]
featureName = list(feature)

for name in featureName:
    feature[name] = np.nan_to_num((feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min()),
                                  0)

features = feature.values
feature_value_list = np.array(features)
labels = data_source.loc[:, 'valencia'].dropna()

print('******************** Entrenamiento comenzado. ***********************')
X = feature_value_list
y = labels
test_size = 0.3
random_seed = 10

X_train, X_test, y_train, y_test = train_test_split(feature_value_list, labels, test_size=test_size,
                                                    random_state=random_seed)

# ==============================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


modelo_5 = MLPRegressor(
    hidden_layer_sizes=(300, 125, 25, 5),
    learning_rate_init=0.01,
    solver='sgd',
    max_iter=1000,
    random_state=54
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo_5.fit(X=X_train, y=y_train)
file_route_3 = 'modelo_5.pkl'
with open(file_route_3, 'wb') as model_file:
    pickle.dump(modelo_5, model_file)


print('******************** Entrenamiento terminado. ***********************')

# ==============================================================================
accuracy_results = []
predictions = modelo_5.predict(X=X_test)
mse = mean_squared_error(y_test, predictions)
print("Error cuadrático medio: {:.2f}".format(mse))
r2 = r2_score(y_test, predictions)
print("Error cuadrático r2: {:.2f}".format(r2))
accuracy_results.append(modelo_5.score(X_test, y_test) * 100)
print('Puntuacion de precision: ', accuracy_results)

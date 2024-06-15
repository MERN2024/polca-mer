# Prueba con atributos dimensionales (activacion)
#
# ==============================================================================
import numpy as np
import pandas as pd
import pickle
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
data_source = pd.read_csv('arousal-test.csv')
feature = data_source.loc[:, 'tempo':]
featureName = list(feature)

for name in featureName:
    feature[name] = np.nan_to_num((feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min()),
                                  0)

features = feature.values
feature_value_list = np.array(features)
labels = data_source.loc[:, 'activacion'].dropna()

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

arousal_test_model_1 = MLPRegressor(
    hidden_layer_sizes=(300, 125, 25, 5),
    learning_rate_init=0.01,
    solver='lbfgs',
    max_iter=1000,
    random_state=123
)

arousal_test_model_2 = MLPRegressor(
    hidden_layer_sizes=(10000, 1000, 100, 10),
    learning_rate_init=0.01,
    solver='lbfgs',
    max_iter=1000,
    random_state=123
)

arousal_test_model_3 = MLPRegressor(
    hidden_layer_sizes=(300, 125, 25, 5),
    learning_rate_init=0.02,
    solver='sgd',
    max_iter=1000,
    random_state=55)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

arousal_test_model_3.fit(X=X_train, y=y_train)
file_route_3 = 'arousal_test_model_3.pkl'
with open(file_route_3, 'wb') as model_file:
    pickle.dump(arousal_test_model_3, model_file)

arousal_test_model_4 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
arousal_test_model_4.fit(X=X_train, y=y_train)
file_route_4 = 'arousal_test_model_4.pkl'
with open(file_route_4, 'wb') as model_file:
    pickle.dump(arousal_test_model_4, model_file)

print('******************** Entrenamiento terminado. ***********************')

# ==============================================================================
accuracy_results = []
predictions = arousal_test_model_3.predict(X=X_test)
mse = mean_squared_error(y_test, predictions)
print("Error cuadrático medio modelo 3: {:.2f}".format(mse))
accuracy_results.append(arousal_test_model_3.score(X_test, y_test) * 100)
print('Puntuacion de precision modelo 3: ', accuracy_results)

accuracy_results = []
predictions = arousal_test_model_4.predict(X=X_test)
mse = mean_squared_error(y_test, predictions)
print("Error cuadrático medio modelo 4: {:.2f}".format(mse))
accuracy_results.append(arousal_test_model_4.score(X_test, y_test) * 100)
print('Puntuacion de precision modelo 4: ', accuracy_results)

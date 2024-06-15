# Prueba con emocion principal
#
# ==============================================================================
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings('ignore')

# ==============================================================================

data_source = pd.read_csv('main-emotion-test.csv')
feature = data_source.loc[:, 'tempo':]
featureName = list(feature)

for name in featureName:
    feature[name] = np.nan_to_num((feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min()),
                                  0)

features = feature.values
feature_value_list = np.array(features)
labels = data_source.loc[:, 'emocion_principal'].dropna()

print('******************** Entrenamiento comenzado. ***********************')
X = feature_value_list
y = labels
test_size = 0.3
random_seed = 7

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

# ==============================================================================
main_emotion_test_model_1 = MLPClassifier(
    hidden_layer_sizes=(20, 20, 20, 20, 20),
    learning_rate_init=0.1,
    solver='adam',
    max_iter=500,
    random_state=123
)

main_emotion_test_model_2 = MLPClassifier(
    hidden_layer_sizes=(20, 20, 20, 20, 20),
    learning_rate_init=0.01,
    solver='lbfgs',
    max_iter=1000,
    random_state=123
)

main_emotion_test_model_1.fit(X=X_train, y=y_train)
file_route_1 = 'main_emotion_test_model_1.pkl'
with open(file_route_1, 'wb') as model_file:
    pickle.dump(main_emotion_test_model_1, model_file)

main_emotion_test_model_2.fit(X=X_train, y=y_train)
file_route_2 = 'main_emotion_test_model_2.pkl'
with open(file_route_2, 'wb') as model_file:
    pickle.dump(main_emotion_test_model_2, model_file)

print('******************** Entrenamiento terminado. ***********************')

# ==============================================================================
accuracy_results = []
predictions = main_emotion_test_model_1.predict(X=X_test)
accuracy_results.append(accuracy_score(predictions, y_test) * 100)
print('Puntuacion de precision modelo 1: ', accuracy_results)

accuracy_results = []
predictions = main_emotion_test_model_2.predict(X=X_test)
accuracy_results.append(accuracy_score(predictions, y_test) * 100)
print('Puntuacion de precision modelo 2: ', accuracy_results)

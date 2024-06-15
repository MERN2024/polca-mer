# Prueba con emociones combinadas reducidas a 4.
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

data_source = pd.read_csv('combined-reduced-test.csv')
feature = data_source.loc[:, 'tempo':]
featureName = list(feature)

for name in featureName:
    feature[name] = np.nan_to_num((feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min()),
                                  0)

features = feature.values
feature_value_list = np.array(features)
labels = data_source.loc[:, 'emocion'].dropna()

print('******************** Entrenamiento comenzado. ***********************')
X = feature_value_list
y = labels

test_size = 0.3
random_seed = 7

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

# ==============================================================================
combined_reduced_test_model = MLPClassifier(
    hidden_layer_sizes=(300, 125, 36),
    learning_rate_init=0.01,
    solver='sgd',
    max_iter=5000,
    tol=0.000100,
    random_state=123456
)
combined_reduced_test_model.fit(X=X_train, y=y_train)
file_route = 'combined_reduced_test_model.pkl'
with open(file_route, 'wb') as model_file:
    pickle.dump(combined_reduced_test_model, model_file)

print('******************** Entrenamiento terminado. ***********************')

# ==============================================================================
accuracy_results = []
predictions = combined_reduced_test_model.predict(X=X_test)
accuracy_results.append(accuracy_score(predictions, y_test) * 100)
print('Puntuacion de precision: ', accuracy_results)

import tensorflow as tf

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def build_regressor(units_1, units_2, optimizer):
    model = Sequential()
    model.add(Dense(units=units_1, activation='relu', input_shape=(X_train_cv.shape[1],)))
    model.add(Dense(units=units_2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Initialize the KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor)

# Define the hyperparameters to tune
param_grid = {'units_1': [3, 5, 10],
              'units_2': [3, 5, 10],
              'optimizer': ['adam', 'rmsprop'],
              'batch_size': [32, 64, 128],
              'epochs': [50, 100, 200]}


def run_hyperparameter_tuning(X_train,y_train):
  # Initialize the GridSearchCV object
  grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

  # Fit the GridSearchCV object on the training data
  grid_search.fit(X_train, y_train)

  # Print the best hyperparameters and the corresponding mean squared error
  print("Best hyperparameters: ", grid_search.best_params_)
  print("Mean squared error: {:.4f}".format(-grid_search.best_score_))


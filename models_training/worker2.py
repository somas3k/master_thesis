import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error

from numpy.random import seed
seed(1234)

tf.random.set_seed(5678)

def worker(params):
    X = params['data'][0]
    y = params['data'][1]
    y_binned = params['data'][2]
    params.pop('data')
    maescores = []
    r2scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=190195)
    for train, test in kfold.split(X, y_binned):
        model = keras.Sequential()
        model.add(layers.Dense(params['units'], activation='tanh', 
                               kernel_regularizer=regularizers.l2(params['kernel']),
                               bias_regularizer=regularizers.l2(params['bias']),
                               activity_regularizer=regularizers.l2(params['activity']),
                               input_shape=[X.shape[1]]))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), metrics=['mae', 'mse'])
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        history = model.fit(X[train], y[train], epochs=params['epochs'],
                              batch_size=params['batch_size'], verbose=0,
                              validation_data=(X[test], y[test]),
                              callbacks=[early_stop])
        train_predictions = model.predict(X[train])
        test_predictions = model.predict(X[test])
        mae_test = mean_absolute_error(y[test], test_predictions)
        maescores.append(mae_test)
        r2_test = r2_score(y[test], test_predictions)
        r2scores.append(r2_test)
#         print("train_max_error: {}".format(max_error(y[train], train_predictions)))
#         print("test_max_error: {}".format(mae_test))
#         print('Train R2 = {:0.2f}.'.format(r2_score(y[train], train_predictions)))
#         print('Test R2 = {:0.2f}.'.format(r2_test))
#     print('kernel: {}, bias: {}, activity: {}'.format(params['kernel'], params['bias'], params['activity']))
#     print('avg mae: {:0.2f}'.format(sum(maescores)/len(maescores)))
#     print('avg r2: {:0.2f}'.format(sum(r2scores)/len(r2scores)))
    return dict({'mae': sum(maescores)/len(maescores), 'r2': sum(r2scores)/len(r2scores)}, **params)
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

def trainer(params):
    X = params['data'][0]
    y = params['data'][1]
    train = params['data'][2]
    test = params['data'][3]
    feature = params['feature']
    units = params['units']
    kernel = params['regularization']['kernel']
    bias = params['regularization']['bias']
    activity = params['regularization']['activity']
    split = params['split']
    model = keras.Sequential()
    model.add(layers.Dense(units, activation='tanh', 
                           kernel_regularizer=regularizers.l2(kernel),
                           bias_regularizer=regularizers.l2(bias),
                           activity_regularizer=regularizers.l2(activity),
                           input_shape=[X.shape[1]]))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mae', 'mse'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit(X[train], y[train], epochs=3000,
                          batch_size=10, verbose=0,
                          validation_data=(X[test], y[test]),
                          callbacks=[early_stop])
    train_predictions = model.predict(X[train])
    test_predictions = model.predict(X[test])
    mae_test = mean_absolute_error(y[test], test_predictions)
    r2_test = r2_score(y[test], test_predictions)
        
    tf.keras.models.save_model(
        model,
        'models_{}\{}_model_{}_{}_{}_{}.h5'.format(feature, split, units, kernel, bias, activity),
        save_format='h5'
    )
    return {
        'kernel': kernel, 
        'bias': bias, 
        'activity': activity, 
        'units': units, 
        'history': history.history,
        'mae': mae_test, 
        'r2': r2_test
    }
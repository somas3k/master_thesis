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
    y_binned = params['data'][2]
    feature = params['feature']
    units = params['units']
    kernel = params['regularization']['kernel']
    bias = params['regularization']['bias']
    activity = params['regularization']['activity']
    maescores = []
    r2scores = []
    best_model = None
    best_score = -1
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=190195)
    for train, test in kfold.split(X, y_binned):
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
        maescores.append(mae_test)
        r2_test = r2_score(y[test], test_predictions)
        r2scores.append(r2_test)
        score = r2_test
        if score > best_score:
            best_model = model
            best_score = score
        
    tf.keras.models.save_model(
        best_model,
        'models_{}\model_{}_{}_{}_{}.h5'.format(feature, units, kernel, bias, activity),
        save_format='h5'
    )
    return {
        'kernel': kernel, 
        'bias': bias, 
        'activity': activity, 
        'units': units, 
        'mae': sum(maescores)/len(maescores), 
        'r2': sum(r2scores)/len(r2scores)
    }
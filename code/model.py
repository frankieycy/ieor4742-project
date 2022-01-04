import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tqdm import tqdm
from data import *

################################################################################

#### Load Data via data.py #####################################################

excess_return, opt_people = DATA['excess_return'], DATA['opt_people']
data_dict, label_dict = prepare_data(excess_return)
one_hot_opt_people = one_hot_dataset(opt_people)

################################################################################

#### Model Functions ###########################################################

def baseline_LSTM():
    '''
    Baseline model 1: LSTM forecast with only time-series data
    '''

    trainX, trainY, testX, testY = timeseries_dataset(data_dict, label_dict)

    model = Sequential()
    model.add(LSTM(200, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=10, batch_size=20, validation_split=0.1, verbose=1)

    return model, history, (trainX, trainY, testX, testY)

def baseline_FNN():
    '''
    Baseline model 2: FNN forecast with only time-series data
    '''

    trainX, trainY, testX, testY = timeseries_dataset(data_dict, label_dict)

    model = Sequential()
    model.add(Dense(200))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=10, batch_size=20, validation_split=0.1, verbose=1)

    data = (trainX, trainY, testX, testY)

    return model, history, data

def model_AltFNN():
    '''
    AltFNN model: FNN forecast with time-series and alt. data
    '''

    trainX, trainY, testX, testY = timeseries_alt_dataset(data_dict, label_dict, one_hot_opt_people)

    model = Sequential()
    model.add(Dense(200))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=10, batch_size=20, validation_split=0.1, verbose=1)

    data = (trainX, trainY, testX, testY)

    return model, history, data

def model_AltReg():
    '''
    AltReg model: Regression forecast with only alt. data
    '''

    one_hot_opt_people_labeled = alt_dataset(data_dict, label_dict, one_hot_opt_people)

    dataX = one_hot_opt_people_labeled.drop('Y', axis=1)
    dataY = one_hot_opt_people_labeled['Y']
    dataX.reset_index(drop=True, inplace=True)
    dataY.reset_index(drop=True, inplace=True)

    n_data = int(dataX.shape[0] * 0.7)
    trainX, trainY = dataX.iloc[:n_data], dataY.iloc[:n_data]
    testX, testY = dataX.iloc[n_data:], dataY.iloc[n_data:]

    trainX = sm.add_constant(trainX)
    model = sm.OLS(trainY, trainX)
    model = model.fit()

    data = (trainX, trainY, testX, testY)

    return model, None, data

def model_LSTMAltReg():
    '''
    LSTM-AltReg model: LSTM forecast with time-series (LSTM) and alt. data (AltReg)
    '''

    trainX, trainY, testX, testY = timeseries_alt_dataset(data_dict, label_dict, one_hot_opt_people)

    alt_dim = one_hot_opt_people.shape[1]
    trainX_ts, testX_ts = trainX[:,:,:-alt_dim], testX[:,:,:-alt_dim]
    trainX_alt, testX_alt = trainX[:,:,-alt_dim:], testX[:,:,-alt_dim:]

    model_ts = Sequential()
    model_ts.add(LSTM(200, input_shape=(trainX_ts.shape[1], trainX_ts.shape[2])))
    model_ts.add(Dense(trainY.shape[1]))
    model_ts.compile(loss='mse', optimizer='adam')
    history_ts = model_ts.fit(trainX_ts, trainY, epochs=10, batch_size=20, validation_split=0.1, verbose=1)

    model_alt, history_alt, data_alt = model_AltReg()

    data_ts = (trainX_ts, trainY, testX_ts, testY)

    return {
        'LSTM': (model_ts, history_ts, data_ts),
        'AltReg': (model_alt, history_alt, data_alt)
    }

def model_LSTMAltFNN():
    '''
    LSTM-AltFNN model: LSTM forecast with time-series (LSTM) and alt. data (AltFNN)
    '''

    trainX, trainY, testX, testY = timeseries_alt_dataset(data_dict, label_dict, one_hot_opt_people)

    alt_dim = one_hot_opt_people.shape[1]
    trainX_ts, testX_ts = trainX[:,:,:-alt_dim], testX[:,:,:-alt_dim]
    trainX_alt, testX_alt = trainX[:,:,-alt_dim:], testX[:,:,-alt_dim:]

    model_ts = Sequential()
    model_ts.add(LSTM(200, input_shape=(trainX_ts.shape[1], trainX_ts.shape[2])))
    model_ts.add(Dense(trainY.shape[1]))
    model_ts.compile(loss='mse', optimizer='adam')
    history_ts = model_ts.fit(trainX_ts, trainY, epochs=10, batch_size=20, validation_split=0.1, verbose=1)

    model_alt = Sequential()
    model_alt.add(Dense(200))
    model_alt.add(Dense(1))
    model_alt.compile(loss='mse', optimizer='adam')
    history_alt = model_alt.fit(trainX, trainY, epochs=10, batch_size=20, validation_split=0.1, verbose=1)

    data_ts = (trainX_ts, trainY, testX_ts, testY)
    data_alt = (trainX, trainY, testX, testY)

    return {
        'LSTM': (model_ts, history_ts, data_ts),
        'AltFNN': (model_alt, history_alt, data_alt)
    }

def plot_train_hist(history):
    '''
    Plot training history, including training and validation loss
    Example:
        model, history, data = baseline_FNN()
        plot_train_hist(history)
    '''

    if history:
        plt.plot(np.log(history.history['loss']), label='train')
        plt.plot(np.log(history.history['val_loss']), label='valid')
        plt.title('training loss')
        plt.legend()
        plt.show()

def plot_pred_result(X, Y, model, type='Train'):
    '''
    Plot prediction results based on model
    Example:
        model, history, data = baseline_FNN()
        trainX, trainY, testX, testY = data
        plot_pred_result(trainX, trainY, model, type='Train')
        plot_pred_result(testX, testY, model, type='Test')
    '''

    yhat = model.predict(X)
    yhat = yhat.reshape(-1)
    yact = Y.reshape(-1)

    rmse = np.sqrt(mean_squared_error(yact, yhat))
    print(f'{type} RMSE: %.4f' % rmse)

    plt.scatter(yact, yhat, s=1, c='k')
    plt.title(f'{type} data')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.show()

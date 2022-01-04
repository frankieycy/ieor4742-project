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
from tqdm import tqdm

################################################################################

DATA_ROOT = 'data/'

PATH_DICT = {
    'categories':       'MF_LargeCap_Categories.parquet',
    'excess_return':    'MF_LargeCap_ExcessReturn_3Y.parquet',
    'fund_size':        'MF_LargeCap_FundSize.parquet',
    'opt_people':       'opt_people.parquet',
}

PATH_DICT = {key: os.path.join(DATA_ROOT, direc) for key, direc in PATH_DICT.items()}

DATA = {key: pd.read_parquet(PATH_DICT[key], engine='pyarrow') for key in PATH_DICT}

################################################################################

#### Time-Series Sampling Parameters ###########################################

# window length between two consecutive features: number of days
feat_window = 90
# performance measures window: number of years
pm_window = 3
lb_window = int(3 * pm_window * 365.25) + 1
# window length between training samples: number of days
sample_window = 30
# test period start
test_start_date = '2020-06-30'

#### Data Functions ############################################################

def prepare_data(data):
    '''
    Generate data_dict and label_dict from excess_return dataframe
    Adopted from dev_prediction.ipynb, from previous groups' work
    Example:
        prepare_data(DATA['excess_return'])
    '''

    data_dict = {ticker: data[ticker].dropna() for ticker in data.columns}

    tickers_to_remove = []

    label_dict = {}
    for ticker, series in tqdm(data_dict.items()):
        if series.isna().sum() == series.shape[0]:
            tickers_to_remove += [ticker]
            continue

        last_date = series.index[-1] - relativedelta(years=pm_window)
        if last_date <= series.index[0]:
            tickers_to_remove.append(ticker)
            continue

        index = series.loc[:series.index[-1] - relativedelta(years=pm_window)].index
        label_dict[ticker] = pd.Series([
            series[date + relativedelta(years=pm_window)] for date in index
        ], index=index)

    _ = [data_dict.pop(ticker) for ticker in tickers_to_remove]

    return data_dict, label_dict

def timeseries_dataset(data_dict, label_dict):
    '''
    Generate time-series training and testing datasets from prepare_data() outputs
    Example:
        data_dict, label_dict = prepare_data(DATA['excess_return'])
        trainX, trainY, testX, testY = timeseries_dataset(data_dict, label_dict)
    '''

    tickers = list(data_dict.keys())

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    # test start date
    checkpoint = datetime.strptime(test_start_date, '%Y-%m-%d') - relativedelta(years=pm_window)

    for ticker in tqdm(tickers):
        label = label_dict[ticker]
        if label.shape[0] == 0:
            continue
        ts = data_dict[ticker].loc[:label.index[-1]]

        indices = [np.arange(i, i+lb_window, feat_window) for i in range(0, ts.shape[0] - lb_window + 1, sample_window)]

        temp_data = np.array([ts.iloc[sub_indices].values for sub_indices in indices])
        if temp_data.shape[0] == 0:
            continue
        temp_labels = np.array([label.loc[ts.index[sub_indices[-1]]] for sub_indices in indices])

        train_indices = [idx for idx in range(temp_data.shape[0]) if ts.index[indices[idx][-1]] <= checkpoint]
        test_indices = [idx for idx in range(temp_data.shape[0]) if ts.index[indices[idx][-1]] > checkpoint]

        train_data += [temp_data[train_indices]]
        train_labels += [temp_labels[train_indices]]

        test_data += [temp_data[test_indices]]
        test_labels += [temp_labels[test_indices]]

    trainX = np.concatenate(train_data)
    trainY = np.concatenate(train_labels)

    testX = np.concatenate(test_data)
    testY = np.concatenate(test_labels)

    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0], 1)

    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
    testY = testY.reshape(testY.shape[0], 1)

    return trainX, trainY, testX, testY

def one_hot(alt_data, col):
    '''
    Cast a feature col of alt_data to one-hot encoded format
    Example:
        opt_people = DATA['opt_people']
        one_hot_auditor = one_hot(opt_people, "Auditor")
    '''

    i = alt_data[col]
    i_df = pd.DataFrame({col: i.tolist()})
    alt_data[col] = labelencoder.fit_transform(alt_data[col])
    enc_df = pd.DataFrame(enc.fit_transform(alt_data[[col]]).toarray())
    enc_df.index = i.index
    return enc_df

def one_hot_dataset(alt_data):
    '''
    Cast all feature cols of alt_data to one-hot encoded format
    Example:
        opt_people = DATA['opt_people']
        one_hot_opt_people = one_hot_dataset(opt_people)
    '''

    enc_df = pd.concat([one_hot(alt_data, col) for col in alt_data.columns], axis=1)
    enc_df.columns = list(range(len(enc_df.columns)))
    return enc_df

def timeseries_alt_dataset(data_dict, label_dict, one_hot_alt_data):
    '''
    Generate time-series & alt-data training and testing datasets from prepare_data() and one_hot_dataset() outputs,
    by concatenating one-hot features to time-series data
    Example:
        opt_people = DATA['opt_people']
        one_hot_opt_people = one_hot_dataset(opt_people)
        data_dict, label_dict = prepare_data(DATA['excess_return'])
        trainX, trainY, testX, testY = timeseries_alt_dataset(data_dict, label_dict, one_hot_opt_people)
    '''

    tickers = list(data_dict.keys())

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    # test start date
    checkpoint = datetime.strptime(test_start_date, '%Y-%m-%d') - relativedelta(years=pm_window)

    for ticker in tqdm(tickers):
        label = label_dict[ticker]
        if label.shape[0] == 0:
            continue
        ts = data_dict[ticker].loc[:label.index[-1]]

        indices = [np.arange(i, i+lb_window, feat_window) for i in range(0, ts.shape[0] - lb_window + 1, sample_window)]

        temp_data = np.array([np.concatenate([ts.iloc[sub_indices].values, one_hot_alt_data.loc[ticker]]) for sub_indices in indices])
        if temp_data.shape[0] == 0:
            continue
        temp_labels = np.array([label.loc[ts.index[sub_indices[-1]]] for sub_indices in indices])

        train_indices = [idx for idx in range(temp_data.shape[0]) if ts.index[indices[idx][-1]] <= checkpoint]
        test_indices = [idx for idx in range(temp_data.shape[0]) if ts.index[indices[idx][-1]] > checkpoint]

        train_data += [temp_data[train_indices]]
        train_labels += [temp_labels[train_indices]]

        test_data += [temp_data[test_indices]]
        test_labels += [temp_labels[test_indices]]

    trainX = np.concatenate(train_data)
    trainY = np.concatenate(train_labels)

    testX = np.concatenate(test_data)
    testY = np.concatenate(test_labels)

    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0], 1)

    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])
    testY = testY.reshape(testY.shape[0], 1)

    return trainX, trainY, testX, testY

def alt_dataset(data_dict, label_dict, one_hot_alt_data):
    '''
    Generate alt-data training and testing datasets from prepare_data() and one_hot_dataset() outputs,
    with mean historical excess returns as labels
    Example:
        opt_people = DATA['opt_people']
        one_hot_opt_people = one_hot_dataset(opt_people)
        data_dict, label_dict = prepare_data(DATA['excess_return'])
        trainX, trainY, testX, testY = alt_dataset(data_dict, label_dict, one_hot_opt_people)
    '''

    tickers = list(data_dict.keys())

    mean_labels = {}

    for ticker in tqdm(tickers):
        label = label_dict[ticker]
        if label.shape[0] == 0:
            continue
        ts = data_dict[ticker].loc[:label.index[-1]]

        indices = [np.arange(i, i+lb_window, feat_window) for i in range(0, ts.shape[0] - lb_window + 1, sample_window)]

        temp_data = np.array([np.concatenate([ts.iloc[sub_indices].values, one_hot_alt_data.loc[ticker]]) for sub_indices in indices])
        if temp_data.shape[0] == 0:
            continue
        temp_labels = np.array([label.loc[ts.index[sub_indices[-1]]] for sub_indices in indices])

        mean_labels[ticker] = temp_labels.mean()

    one_hot_alt_data_labeled = one_hot_alt_data.loc[list(mean_labels.keys())]
    one_hot_alt_data_labeled['Y'] = list(mean_labels.values())

    return one_hot_alt_data_labeled

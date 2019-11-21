import numpy as np
import pandas as pd

from datetime import datetime
import itertools as it

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# from fbprophet import Prophet

import math

import matplotlib.pyplot as plt


def last_value(train, test):
    last_value = train[-1]
    yhat = [last_value] * test.shape[0]
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'last_value', 
            'mse': mse,
            'rmse': rmse},last_value

def simple_avg(train, test):
    simple_average = train.mean()
    yhat = [simple_average] * test.shape[0]
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'simple_avg', 
            'mse': mse,
            'rmse': rmse},simple_average    

def moving_avg(train, test, interval = 30):
    moving_average = train.rolling(interval).mean()[-1]
    yhat = [moving_average] * test.shape[0]
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'moving_avg', 
            'mse': mse,
            'rmse': rmse},moving_average

def linear_holt(train,test):
    lin_h = Holt(train).fit()
    yhat = lin_h.forecast(test.shape[0])
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'holt', 
            'mse': mse,
            'rmse': rmse},yhat

# def prophet(train,test):
#     mod_train = pd.DataFrame(train)
#     mod_test = pd.DataFrame(test)

#     mod_train['y'] = train
#     mod_test['y'] = test
    
    mod_train['ds'] = train.index 
    mod_test['ds'] = test.index

    model = Prophet()
    model.fit(mod_train)
    yhat = model.predict(mod_test).yhat

    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'prophet': 'moving_avg', 
            'mse': mse,
            'rmse': rmse}




#     model = Prophet()
#     model.fit(mod_train)
#     yhat = model.predict(mod_test).yhat

#     mse = mean_squared_error(test, yhat)
#     rmse = mse**.5
#     return {'prophet': 'moving_avg', 
#             'mse': mse,
#             'rmse': rmse}
#     pass


def run_models(train,test, interval = 30):
    model_strength = pd.DataFrame(columns=['model_type', 'mse', 'rmse'])

    row,_ = last_value(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    row,_ = simple_avg(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    row,_ = moving_avg(train,test, interval = interval)
    model_strength = model_strength.append(row, ignore_index= True)

    row,_ = linear_holt(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    # row = prophet(train,test)
    # model_strength = model_strength.append(row, ignore_index= True)

    return model_strength

def plot_figures(train,test, interval = 30, name = 'Predictions'):
    _,last_value_yhat = last_value(train,test)
    _,simple_avg_yhat = simple_avg(train,test)
    _,moving_avg_yhat = moving_avg(train,test, interval = interval)
    _,holt_yhat = linear_holt(train,test)
    
    train = train.append(test[:1])
    plt.figure(figsize=(14,8))
    x = plt.plot(train['2018-09':])
    y = plt.plot(test, color = 'firebrick')
    a = plt.hlines(last_value_yhat, xmin = test.index.tolist()[0], xmax= test.index.tolist()[-1], linestyles=':')
    b = plt.hlines(simple_avg_yhat, xmin = test.index.tolist()[0], xmax= test.index.tolist()[-1], linestyles='--')
    c = plt.hlines(moving_avg_yhat, xmin = test.index.tolist()[0], xmax= test.index.tolist()[-1], linestyles='-')
    d = plt.plot(holt_yhat, '-.', c = 'black')

    plt.legend([x,y,a,b,c,d], labels = ['Train', 'Actual', 'Holt', 'Last Value', 'Simple Average', 'Moving Average'])
    plt.title(name)
    plt.show()

def final_predictions(train,test):
    df = train.append(test)
    predicted = pd.DataFrame(index = pd.date_range('2018-12-07', '2018-12-20', freq='d'))

    for col in df.columns:
        models = run_models(train[col], test[col], interval = 4)
        mod_type = models[models.rmse == models.rmse.min()].model_type
        predicted[col] = 0
        if mod_type.index[0] == 0:
            predicted[col] = last_value(df[col],predicted[col])[1]
        elif mod_type.index[0] == 1:
            predicted[col] = simple_avg(df[col],predicted[col])[1]
        elif mod_type.index[0] == 2:
            predicted[col] = moving_avg(df[col],predicted[col])[1]
    return df.append(predicted)
    
    
    pass
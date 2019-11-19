import numpy as np
import pandas as pd

from datetime import datetime
import itertools as it

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from fbprophet import Prophet

import math

import matplotlib.pyplot as plt


def last_value(train, test):
    last_value = train[-1]
    yhat = [last_value] * test.shape[0]
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'last_value', 
            'mse': mse,
            'rmse': rmse}

def simple_avg(train, test):
    simple_average = train.mean()
    yhat = [simple_average] * test.shape[0]
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'simple_avg', 
            'mse': mse,
            'rmse': rmse}

def moving_avg(train, test):
    moving_average = train.rolling(30).mean()[-1]
    yhat = [moving_average] * test.shape[0]
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'moving_avg', 
            'mse': mse,
            'rmse': rmse}

def linear_holt(train,test):
    lin_h = Holt(train).fit()
    yhat = lin_h.forecast(test.shape[0])
    mse = mean_squared_error(test, yhat)
    rmse = mse**.5
    return {'model_type': 'holt', 
            'mse': mse,
            'rmse': rmse}

def prophet(train,test):
    mod_train = pd.DataFrame(train)
    mod_test = pd.DataFrame(test)

    mod_train['y'] = train
    mod_test['y'] = test
    
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
    pass







def run_models(train,test):
    model_strength = pd.DataFrame(columns=['model_type', 'mse', 'rmse'])

    row = last_value(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    row = simple_avg(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    row = moving_avg(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    row = linear_holt(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    row = prophet(train,test)
    model_strength = model_strength.append(row, ignore_index= True)

    return model_strength
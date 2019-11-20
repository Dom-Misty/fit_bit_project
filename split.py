import pandas as pd 
import numpy as np 

def split_daily(df):
    train = df[:'2018-11-22'].resample("D").agg("mean")
    test = df['2018-11-23':].resample("D").agg("mean")
    return train, test

def split_weekly(df):
    weekly = df.resample('w').mean()
    train = weekly[:'2018-11-25']
    test = weekly['2018-12-02':]
    return train, test

import pandas as pd 
import numpy as np 

def split_daily(df):
    train = df[:'2018-11-22'].resample("D").agg("mean")
    test = df['2018-11-23':].resample("D").agg("mean")
    return train, test

def split_weekly(df):
    train = df[:'2018-11-25'].resample("W").agg("mean")
    test = df['2018-12-02':].resample("W").agg("mean")
    return train, test

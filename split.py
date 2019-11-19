import pandas as pd 
import numpy as np 

def split_two_weeks(df, resample):
    train = df[:'2018-11-22'].resample(resample).agg(sum)
    test = df['2018-11-23':].resample(resample).agg(sum)
    return train, test

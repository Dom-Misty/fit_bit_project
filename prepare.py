import pandas as pd 
import numpy as np 

def rename_cols(df):
    df = df.rename(columns={"Date":"date","Calories Burned":"cal_burned", "Steps":"steps", "Distance":"distance", "Floors":"floors",
                  "Minutes Sedentary":"mins_sedentary", "Minutes Lightly Active":"mins_light_activity",
                   "Minutes Fairly Active":"mins_fair_activity","Minutes Very Active":"mins_very_active",
                   "Activity Calories":"activity_cals"})
    return df

def set_date_index(df):
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace = True)
    df.sort_index(inplace=True)
    return df

def fix_object_columns(df):
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.replace(",","").astype("float")
    return df

def fix_outliers(df):
    df.loc["2018-08-26", "cal_burned"] = 2144
    return df

def make_float(df):
    df = df.astype("float")
    return df

def prep_fitbit_data(df):
    df = rename_cols(df)
    
    df = set_date_index(df)

    df = fix_object_columns(df)

    df = fix_outliers(df)

    df = make_float(df)

    return df

def add_features(df):
    df["mins_total_activity"] = df.mins_light_activity + df.mins_fair_activity + df.mins_very_active
    df['day'] = df.index.strftime('%w-%a')
    return df   

def acquire_data():
    return pd.read_csv('activity_log.csv')
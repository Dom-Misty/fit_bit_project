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
    return df

def fix_object_columns(df):
    for col in df.columns:
        if df[col].dtype.str == '|O':
            df[col] = df[col].str.replace(",","_").astype(float)
    return df

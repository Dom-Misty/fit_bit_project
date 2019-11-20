import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def time_series_graphs(df): 
    for col in df.columns:
        plt.plot(df[col])
        plt.title(col)
        plt.show()

def calc_height(df):
    #find average steps per mile and compute height
    spm = (df.steps/df.distance).mean()
    height = (5280*12)/(.413*spm)
    return height

def subject_profile(df):
    height = calc_height(df)

    bmr = df.cal_burned['2018-06-26']
    #equation for age and weight
    age = np.linspace(15,80,100)
    weight_m = (bmr -66 - 12.7*height + 6.76*age)/6.2
    male = plt.plot(age,weight_m, c = 'navy')

    weight_f = (bmr - 655.1 - 4.7*height + 4.7*age)/4.35
    female = plt.plot(age,weight_f, c = 'pink')

    plt.title('Subject Profile')
    plt.xlabel('age')
    plt.ylabel('weight')
    plt.legend([male,female], labels = ['Male','Female'])
    plt.show()
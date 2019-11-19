import matplotlib.pyplot as plt
import pandas as pd

def time_series_graphs(df): 
    for col in df.columns:
        plt.plot(df[col])
        plt.title(col)
        plt.show()
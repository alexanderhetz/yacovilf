
# Yet Another CoViD LSTM Forecast

print('Loading Packages')
print('\n')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz
from IPython.display import Image

print('Loading data...')
print('\n')

# Download data into a pandas dataframe
# This allows to get updated data
# Number of infected, number of recovered and number of deaths
covinft=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
covrect=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
covdeat=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

# Get a numpy array of latitudes as a mean of latitudes by region
lat=pd.concat([covinft['Country/Region'],covinft['Lat']],axis=1).groupby(['Country/Region']).mean().T.to_numpy(copy=True)

# Get rid of unwanted data
del(covinft['Lat'])
del(covinft['Long'])
del(covinft['Province/State'])
del(covrect['Lat'])
del(covrect['Long'])
del(covrect['Province/State'])
del(covdeat['Lat'])
del(covdeat['Long'])
del(covdeat['Province/State'])

# Add cases by region to get total cases by country
covinfd=covinft.groupby(['Country/Region']).sum().T
covrecd=covrect.groupby(['Country/Region']).sum().T
covdead=covdeat.groupby(['Country/Region']).sum().T

# Dates and names of countries
index=np.array(covinfd.index)
columns=np.array(covrecd.columns)

# Infected, recovered, and deaths
# where each column of the matrices represents a country
# and each row represents a date
covinf=covinfd.to_numpy(copy=True)
covrec=covrecd.to_numpy(copy=True)
covdea=covdead.to_numpy(copy=True)

# Show one every seven dates
ticks=np.copy(index)
for i in range(0,len(index)):
    if i%7!=0:
        ticks[i]=''

# Get number of column given name of country
def countrynumber(country):
    return np.where(columns==country)[0][0]   
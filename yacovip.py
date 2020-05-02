print('Loading packages...')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('Downloading data...')
print('\n')

# Download data into a pandas dataframe
# This allows to get updated data
covinft=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
covrect=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

# Get rid of unwanted data
del(covinft['Lat'])
del(covinft['Long'])
del(covinft['Province/State'])
del(covrect['Lat'])
del(covrect['Long'])
del(covrect['Province/State'])

# Add cases by region to get total cases by country
covinfd=covinft.groupby(['Country/Region']).sum().T
covrecd=covrect.groupby(['Country/Region']).sum().T

# Create numpy arrays with the data, because I have to learn that first
# Dates and names of countries
index=np.array(covinfd.index)
columns=np.array(covrecd.columns)

# Infected and recovered
# where each column of the matrices represents a country
covinf=covinfd.to_numpy(copy=True)
covrec=covrecd.to_numpy(copy=True)

# Show one every seven dates
ticks=np.copy(index)
for i in range(0,len(index)):
    if i%7!=0:
        ticks[i]=''

# Get number of column given name of country
def countrynumber(country):
    return np.where(columns==country)[0][0]        

def numberplot2(n,m):
    
    # Calculate r and set it to zero if denominator is zero
    d=covinf[:len(covinf)-m,n]-covrec[:len(covrec)-m,n]
    r=np.zeros(len(covinf)-m)
    for i in range(0,len(d)):
        if d[i]!=0:
            r[i]=(covinf[m:,n]-covinf[:len(covinf)-m,n])[i]/d[i]
        else:
            r[i]=0
            
    # Plot the plots
    fig,(gr1,gr2)=plt.subplots(1,2,figsize=(15,6))
    fig.suptitle(columns[n],fontsize='18')
    
    gr1.plot(index[:len(index)-7],r,'m',np.ones(len(index)-7),'--')
    gr1.set_xticks(index[:len(index)-7])
    gr1.set_xticklabels(ticks[:len(index)-7], rotation=90)
    gr1.grid(axis='both',linestyle='--',which='both')
    gr1.set_xlabel('Date',fontsize='14')
    gr1.legend(('r', 'r = 1'),
              loc='upper left', shadow=True, fontsize='14')
    
    gr2.plot(index,covrec[:,n],'b',covinf[:,n],'r')
    gr2.set_xticks(index)
    gr2.set_xticklabels(ticks, rotation=90)
    gr2.set_yscale('log')
    gr2.grid(axis='both',linestyle='--',which='both')
    gr2.set_xlabel('Date',fontsize='14')
    gr2.legend(('Recovered', 'Infected'),
              loc='upper left', shadow=True)
    
    plt.show()
    
# Function composition to get plots from country name
def countryplot2(country,m):
    numberplot2(countrynumber(country),m)

# User interface
print('Yet Another CoViD Plot')
print('\n')
print('r is a guesstimation of how many people every infected person infects.')
print('r equals the newly infected of next week divided by the difference \nbetween total number of infected and the recovered ones.')
print('If infected(t) = exp(a*t), then r = exp(7*a).')
print('\n')
print('\"Countries\" for a list of countries')
print('\"Exit\" to exit')
print('\n')

respuesta=input('Which country? ');

while respuesta!='Exit':
    if respuesta=='Countries':
        print('\n')
        print(columns)
    elif np.isin(respuesta,columns):
         countryplot2(respuesta,7)
    else:
        print('Undefined command')
    respuesta=input('Which country? ');

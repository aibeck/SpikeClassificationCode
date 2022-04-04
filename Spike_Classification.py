 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:53:56 2020

@author: ABeck
"""

import pandas as pd
import numpy as np
import datetime as dt
import glob
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def getScaling():
    # Makine scaling method, based on a random dataset
    scaledata = (pd.read_csv(training_files[1]))
    scaledata.drop(scaledata.columns[[0, scaledata.columns.get_loc("Epoch2")]], axis=1, inplace=True)
    columnNames = scaledata.columns
    scaledata = scaledata.values;
    scaledata = scaledata[:, :98]
    
    hour = [dt.datetime.strptime(i, '%m/%d/%y %H:%M').time().hour for i in np.array(scaledata[:,95]).tolist()];
    day = [dt.datetime.strptime(i, '%m/%d/%y %H:%M').date().day for i in np.array(scaledata[:,95]).tolist()];
    scaledata[:,95] = list(np.array(hour) + (np.array(day) - 24)*24)
    
    scaledataLights = np.array([1 if i >= 6 and i <= 17 else 0 for i in scaledata[:,95]])
    scaledata = np.append(scaledata, scaledataLights[:, None], axis = 1)
    scaling = MinMaxScaler().fit(scaledata[:, np.r_[0:96, 98:99]])
    
    return columnNames, scaling

def formatData(data_read, byDay):
    data = []
    dayList = []
    mouseModel = []
    
    for name in data_read:  
        fulldata = (pd.read_csv(name))
        fulldata["time_stamps"] = pd.to_datetime(fulldata["time_stamps"], errors='coerce')
        fulldata["time_stamps"] = fulldata["time_stamps"].dt.strftime("%m/%d/%y %H:%M")
        fulldata.drop(fulldata.columns[[0, fulldata.columns.get_loc("Epoch2")]], axis=1, inplace=True)
        fulldata = fulldata.values;
        #fulldata = fulldata[:, :98]
        fulldata = fulldata[:, np.r_[:96, 98:100]]
        
        hour = [dt.datetime.strptime(i, '%m/%d/%y %H:%M').time().hour for i in np.array(fulldata[:,95]).tolist()];
        month = [dt.datetime.strptime(i, '%m/%d/%y %H:%M').date().month for i in np.array(fulldata[:,95]).tolist()];
        day = [dt.datetime.strptime(i, '%m/%d/%y %H:%M').date().day for i in np.array(fulldata[:,95]).tolist()];
            
        day = list((np.array(month)*30) + np.array(day))
        
        if name[:3] == "app":
            day = [x - 88 for x in day]
        elif name[:3] == "fad":
            day = [x - 173 for x in day]
        
        lights = np.array([1 if i >= 6 and i <= 17 else 0 for i in hour])
        fulldata[:,95] = [dt.datetime.strptime(i, '%m/%d/%y %H:%M').time().hour for i in np.array(fulldata[:,95]).tolist()];
        
        fulldata = np.append(fulldata, lights[:, None], axis = 1)
        data.extend(fulldata)
        dayList.extend(day)
        
        model = [name[4:8]]*len(fulldata)
        mouseModel.extend(model)
        
    data = np.array(data, dtype=object)
    
    X = data[:, np.r_[0:95, 98:99]]
    hour = list(data[:, 95])
    
    Y = data[:, 96:98]
    Ybinary = [1 if x == 'Label' else 0 for x in Y[:, 0]];
        
    return X, list(Y[:,1]), Ybinary, dayList, hour, mouseModel

def loadMouse(mouseNum):
    
    files = glob.glob("*{}*.csv".format(mouseNum))
    
    if not files:
        print("Try again")
        return
    
    X, Yspike, Ylabel, day, hour, mouseModel = formatData(files, True)
    
    return X, Yspike, Ylabel, day, hour, mouseModel

ML = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
  
#Imports and formats training data
os.chdir('G:/My Drive/Alzheimers Spike Analyses/exports/TrainAPP') #Change folder for training
training_files = glob.glob('*.csv')
    
columnNames, scaling = getScaling()
    
#Loads training data and trains algorithm
data_read_train = training_files;
X_train, Yspike_train, Ylabel_train, day_train, hour_train, model_train = formatData(data_read_train, False)

ML.fit(X_train, Ylabel_train)

os.chdir('G:/My Drive/Alzheimers Spike Analyses/exports') #Change folder for analysis

# Predicted number of spikes by hour
#   'app' for APP/PS!; 'fad' for 5xFAD
os.chdir('G:/My Drive/Alzheimers Spike Analyses/exports/forGraphs') #Change folder for analysis
mouseNum = 'APP'
X_mouse, Yspike_mouse, Ylabel_mouse, day_mouse, hour_mouse, model_mouse = loadMouse(mouseNum)

predictions_mouse = ML.predict(X_mouse)

eventData = pd.DataFrame({'mouseNum' : model_mouse, 'dayFromTamoxifen' : day_mouse, 'hour' : [x + y*24 for x, y in zip(hour_mouse, day_mouse)], 'predictedSpike' : predictions_mouse})

#format data for plotting
eventCount = eventData.groupby(['mouseNum', 'dayFromTamoxifen', 'hour']) #groups data by mouseNum, day, and hour
events = eventCount.agg(['mean', 'std']) #calculates mean and sd by day and lightsON
#events.drop(events.columns[[0, 1]], axis=1, inplace = True) #removes unnecessary columns
events.columns = ['mean', 'std'] #renames columns for easier access
events["mean"] = 100 * events["mean"] #multiply to get value into percentage
events["std"] = 100 * events["std"] #ditto
events = events.reset_index() #brings day and lightsON features out of index for use

events_shrt = events[events['dayFromTamoxifen'].isin([-7, 7, 14, 21])]

#events.plot.line()
#events_shrt.plot.scatter(x='dayFromTamoxifen', y='mean', figsize=(11,5), title = "Mouse {}".format(mouseNum)) #scatter plot
#plt.xlabel('Day From Tamoxifen', fontsize=18) #changes x label
#plt.ylabel('Predicted Spike %', fontsize=18) #changes y label

#Mixed Effects Linear Regression
import statsmodels.formula.api as smf
LRmodel = smf.mixedlm("mean ~ hour", events_shrt, groups=events_shrt["mouseNum"]).fit()
print(LRmodel.summary())
print(LRmodel.params["hour"])
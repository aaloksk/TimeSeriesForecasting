# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 07:10:20 2023

@author: Aalok
"""


import os
import pandas as pd
from pykalman import KalmanFilter
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

#Setting working directory
path = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\TimeSeriesForecasting'
os.chdir(path)


#J17 data
#df1 = pd.read_csv('j17_imp.csv')


#Comal SPrings data
df1 = pd.read_csv('Comal_imp.csv')


#San Marcos SPrings data
#df3 = pd.read_csv('SanMarcos_imp.csv')


#Extracting data by setting date as an index
df1.index = pd.to_datetime(df1['date'], format='%m/%d/%Y')
df1 = df1.iloc[:,1]
df1 = pd.DataFrame(df1)
df1

# create a new dataframe with complete date range to check for missing data
date_range = pd.date_range(start='1927-12-19', end='2023-04-16', freq='D')
df1_full = pd.DataFrame(index=date_range, columns=['Dummy'])

#If missing
# merge the two dataframes to add missing dates
#df1_new = pd.merge(df1, df1_full, left_index=True, right_index=True, how='outer')
#df1_new = df1_new.iloc[:,0]

#Water Level Elevation Data extraction and visualisation
Q = df1['Q']
Q.plot()
Qmax = max(Q)
Q = Q / max(Q)

#Now
#Data should be in following format with X as input and Y as input
#Use of 4 data to predict 5th data
#X Y
#[[[Jan1], [Jan2], [Jan3], [Jan4]]] [Jan5]
#[[[Jan2], [Jan3], [Jan4], [Jan5]]] [Jan6]
#[[[Jan3], [Jan4], [Jan5], [Jan6]]] [Jan7]
#Functing for this arrangement
def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

#Setting number of lags
WINDOW_SIZE = 5

#Input Output Arrangement b  calling the function
X1, y1 = df_to_X_y(Q, WINDOW_SIZE)

#CHecking the shape of the formatting
X1.shape, y1.shape

#Splitting the data to train, test and validation split
X_train1, y_train1 = X1[:26300], y1[:26300]
X_val1, y_val1 = X1[26300:29680], y1[26300:29680]
X_test1, y_test1 = X1[29680:], y1[29680:]

#Check shape of splits
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

#Model Architecture
model1 = Sequential() #Model Stacking INitialization
model1.add(InputLayer((5, 1))) #Input layer
model1.add(LSTM(128)) #LSTM Layer 
model1.add(Dense(8, 'relu')) #Dense layer with relu activation function
model1.add(Dense(1)) #Output layer
model1.summary() #Printing model summary

#Model Compilation
cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.00001), metrics=[RootMeanSquaredError()])

#FItting the model using training data
model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])

#Loading the best model based on loss function
model1 = load_model('model1/')
#######################################################

#Train data prediction comparision
train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results = train_results * Qmax
train_results


#xlabel = 'Days from 1932-11-12', ylabel = 'Water Level Elevation (ft)'
#Plot for training data
plt.plot(train_results['Train Predictions'], label='Predictions')
plt.plot(train_results['Actuals'], label='Actuals')
plt.xlabel('Days from 1927-12-19')
plt.ylabel('Discharge (cfs)')
plt.legend()

#Plot for training data zoomed in
plt.plot(train_results['Train Predictions'][500:1000], label='Predictions')
plt.plot(train_results['Actuals'][500:1000], label='Actuals')
plt.xlabel('Days from 1927-12-19')
plt.ylabel('Discharge (cfs)')
plt.legend()

#Plot predictions versus actuals
plt.scatter(train_results['Train Predictions'], train_results['Actuals'])
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.plot(range(50,500), range(50,500), color='red') #Add red line for x=y
plt.show()

#Validation Data Result
val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results = val_results * Qmax
val_results

plt.plot(val_results['Val Predictions'][50:200], label='Predictions')
plt.plot(val_results['Actuals'][50:200], label='Actuals')
plt.xlabel('Days')
plt.ylabel('Discharge (cfs)')
plt.legend()

#Plot predictions versus actuals
plt.scatter(val_results['Val Predictions'], val_results['Actuals'])
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.plot(range(150,500), range(150,500), color='red') #Add red line for x=y
plt.show()

#Test Data Result
test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results = test_results * Qmax
test_results

plt.plot(test_results['Test Predictions'][50:500], label='Predictions')
plt.plot(test_results['Actuals'][50:500], label='Actuals')
plt.xlabel('Days')
plt.ylabel('Discharge (cfs)')
plt.legend()

#Plot predictions versus actuals
plt.scatter(test_results['Test Predictions'], test_results['Actuals'])
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.plot(range(100,450), range(100,450), color='red') #Add red line for x=y
plt.show()

#RMSEs
np.sqrt(np.mean((train_results['Train Predictions'] - train_results['Actuals'])**2))
np.sqrt(np.mean((val_results['Val Predictions'] - val_results['Actuals'])**2))
np.sqrt(np.mean((test_results['Test Predictions'] - test_results['Actuals'])**2))

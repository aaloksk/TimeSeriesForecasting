# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:34:37 2023

@author: Aalok
"""

import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#Setting working directory
path = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\TimeSeriesForecasting\\Exogenous'
os.chdir(path)


# load data
df = pd.read_csv('all_var.csv')

# define variables and lags
variables = ['PPT_mm', 'Tmean_deg', 'Tdmean_deg', 'WaterLevelElevation']
lags = [1, 2, 3]

# create lagged dataset
data = pd.DataFrame()
for i in range(len(variables)):
    var = variables[i]
    for j in lags:
        colname = var + '_lag' + str(j)
        data[colname] = df[var].shift(j)
data = data.dropna()

# split data into train and test sets
train_size = int(len(data) * 0.7)
train, test = data.iloc[0:train_size,:], data.iloc[train_size:len(data),:]

# reshape input to be 3D [samples, timesteps, features]
X_train = np.reshape(train.iloc[:,0:len(lags)*len(variables)].values, 
                     (train.shape[0], len(lags), len(variables)))
X_test = np.reshape(test.iloc[:,0:len(lags)*len(variables)].values, 
                    (test.shape[0], len(lags), len(variables)))
y_train = train.iloc[:,-1].values
y_test = test.iloc[:,-1].values

# normalize input variables by their maximum value
X_train_max = X_train.max(axis=0)
X_train /= X_train_max
X_test /= X_train_max

# normalize y_train and y_test by their maximum values
y_train_max = y_train.max()
y_train /= y_train_max
y_test /= y_train_max

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(8, 'relu')) #Dense layer with relu activation function
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(X_train, y_train, epochs=10, batch_size=24, 
                    validation_data=(X_test, y_test), verbose=2, shuffle=False)

# make predictions
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

# evaluate model performance
train_rmse = np.sqrt(np.mean((y_train - yhat_train.squeeze())**2))
test_rmse = np.sqrt(np.mean((y_test - yhat_test.squeeze())**2))
print('Train RMSE: %.3f' % train_rmse)
print('Test RMSE: %.3f' % test_rmse)

# plot actual vs predicted for train and test datasets
plt.plot(y_train, label='actual_train')
plt.plot(yhat_train.squeeze(), label='predicted_train')
plt.plot([None for i in y_train] + [x for x in y_test], label='actual_test')
plt.plot([None for i in y_train] + [x for x in yhat_test.squeeze()], label='predicted_test')
plt.legend()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:39:45 2017

@author: tulincakmak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
training_set=pd.read_csv('Google_Stock_Price_Train.csv')0
training_set=training_set.iloc[:,1:2].values

#feature scaling (making data normalisation)
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
training_set=sc.fit_transform(training_set)

#getting the inputs and the outputs
X_train=training_set[0:1257]
y_train=training_set[1:1258]

#reshaping
X_train=np.reshape(X_train, (1257, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initializing the RNN
regressor=Sequential()

#adding the input layer and the LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

#adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size=32 ,epochs=200)

#getting the real stock price of 2017
test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values

#getting the predicted stock price of 2017
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs, (20, 1, 1))
predicted_stock_price=regressor.predict(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Goggle Stock Price')
plt.xlabel('Time')
plt.ylabel('Goggle Stock Price')
plt.legend()
plt.show()



#homework

#getting the real stock price of 2012-2016
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

#getting the predicted stock price of 2012-2016
predicted_stock_price_train=regressor.predict(X_train)
predicted_stock_price_train=sc.inverse_transform(predicted_stock_price_train)

#visualizing the results
plt.plot(real_stock_price_train, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
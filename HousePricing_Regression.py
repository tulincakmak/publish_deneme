# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:21:33 2018

@author: tulincakmak
"""



import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# load dataset
data = pd.read_csv("boston.csv")

test=pd.read_csv("TestHouse.csv")

test=test.iloc[:,:].as_matrix()

LABEL='MV'


def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label

x_train, y_train, x_test, y_test = split_data(data, 0.20, LABEL)
input_dim=len(x_train.values[0, :])

x_train=x_train.iloc[:,:].as_matrix()

x_test=x_test.iloc[:,:].as_matrix()



def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    
def wider_model():
	# create model
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.fit(x_train, y_train, batch_size = 25, epochs = 200, validation_data=(x_test, y_test), validation_split=0.2)
    return model

def fit_model(model):
    model.fit(x_train, y_train, batch_size = 25, epochs = 200, validation_data=(x_test, y_test), validation_split=0.2)
    prediction=model.predict(x_test)    #X_test yerine predict ettirilcek dosyayı yaz!!
    print(prediction)

 
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))



fit_model(wider_model()) 
fit_model(baseline_model()) 
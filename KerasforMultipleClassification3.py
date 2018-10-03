# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:48:22 2017

@author: tulincakmak
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("iris_dataset.csv")

LABEL='kolon5'

def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label


x_train, y_train, x_test, y_test = split_data(dataframe, 0.20, LABEL)

input_dim=len((x_train.columns))
output_dim=len(y_train.unique())

#Bu kısım gerekli olmayacak çünkü datalar zaten düzgün bir şekilde gelmiş olcak!!!!
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)

encoder.fit(y_test)
encoded_Y_test = encoder.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)


#keras expects model inputs to be numpy arrays - not pandas.DataFrame
x_train=x_train.iloc[:,:].as_matrix()

x_test=x_test.iloc[:,:].as_matrix()

# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, x_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))






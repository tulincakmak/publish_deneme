# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:12:53 2017

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:09:05 2017

@author: tulincakmak
"""

import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

LABEL = "Exited"
dataset.drop("Surname", axis=1, inplace=True)
dataset.drop("RowNumber", axis=1, inplace=True)
dataset.drop("CustomerId", axis=1, inplace=True)
dataset.drop("Geography", axis=1, inplace=True)
dataset.drop("Gender", axis=1, inplace=True)

def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

x_train, y_train, x_test, y_test = split_data(dataset, 0.20, LABEL)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


input_dim=len(x_train[0, :])


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 25, epochs = 500)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)

ev=classifier.evaluate(x_train,y_train)

y_pred2=[]

i=0
for  k in y_pred:
    print(y_pred[i][0])
    t=y_pred[i][0]
    y_pred2.append(t)
    i=i+1

y_pred2=pd.DataFrame(data=y_pred2)
print(y_pred2.round())
   
classifier.save('my_model.h5')  #BU KISIMDA MODELİ KAYDEDİYOR DOSYA UZANTISI H5 OLMALI BUNUN İÇİNDE ANACONDADANIN ENVİRONMENT ında DA YÜKLÜ OLMALI!!!
from keras.models import load_model
model=load_model('my_model.h5')
t = model.predict(x_test, batch_size=10, verbose=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2.round())


print('prediction = %s '% y_pred)
print('accuracy = %s '% ev[1])
print('mean = %s '% ev[0])
print(cm)



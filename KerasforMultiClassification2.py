# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:24:39 2017

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:37:57 2017

@author: tulincakmak
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)


# load dataset
dataset = pd.read_csv('iris_dataset.csv')



LABEL='kolon5'

def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label


x_train, y_train, x_test, y_test = split_data(dataset, 0.20, LABEL)

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

model = Sequential()
model.add(Dense(8, input_dim=input_dim, activation='relu'))
model.add(Dense(8, input_dim=input_dim, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, dummy_y, epochs=100, batch_size=10, verbose=0)

model.save('my_model.h5')  #BU KISIMDA MODELİ KAYDEDİYOR DOSYA UZANTISI H5 OLMALI BUNUN İÇİNDE ANACONDADANIN ENVİRONMENT ında DA YÜKLÜ OLMALI!!!
from keras.models import load_model
model=load_model('my_model.h5')

model.evaluate(x_test, dummy_y_test, verbose=0)

y_pred=model.predict(x_test)


y_pred2=[]

#reverse of np_utils.to_categorical
i=0
for  k in y_pred:
    print(y_pred[i].round())
    t=y_pred[i].round()
    y_pred2.append(t)
    i=i+1

#reverse of LabelEncoder
t2=0
for t in y_pred2:
    k=np.argmax(y_pred2[t2])
    print(encoder.inverse_transform(k))
    t2=t2+1


ev=model.evaluate(x_test, dummy_y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], ev[1]*100))
print('ev = %s '% ev)





# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:16:13 2018

@author: tulincakmak
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler



data=pd.read_csv('aab.csv', nrows=1000)

scaler=MinMaxScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_df, columns=data.columns)
data=scaled_df


y=data['y']
X=data.drop(['y'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


svr_grid = [
  {'C': [.1,1,10,50,100,500,1000], 'kernel': ['linear']},
  {'C': [.1,1,10,50,100,500,1000], 'gamma': [0.001, 0.0001, 'auto'], 'kernel': ['rbf']},
#  {'C': [.1,1,10,50,100,500,1000], 'gamma': [0.001, 0.0001, 'auto'], 'kernel': ['tan']}, ##olmicak muhtemelen
  {'C': [.1,1,10,50,100,500,1000], 'gamma': [0.001, 0.0001, 'auto'], 'kernel': ['sigmoid']},
  {'C': [.1,1,10,50,100,500,1000], 'gamma': [0.001, 0.0001, 'auto'], 'kernel': ['poly'], 'degree':[2,3,4]}
 ]
model=SVR()
clf = GridSearchCV(model, svr_grid , cv=5, n_jobs=2,
                       scoring='r2')


clf.fit(X_train, y_train)

clf.best_params_
clf.best_score_
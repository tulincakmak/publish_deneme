# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:32:17 2018

@author: tulincakmak
"""


import pandas as pd
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy import stats
import h2o
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae

data=pd.read_excel('everthing_avarage2_traindata.xlsx')

data['bölge'] = data['bölge'].str.strip()
data['booking_value_index'] = data['booking_value_index'].str.strip()

data.groupby('bölge')['bölge'].count()
data['bölge']=data['bölge'].fillna('Marmara')



data["bölge"]=data["bölge"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))


mapping2={'Marmara': 'Marmara', 'Ege': 'Ege', 'Akdeniz':'Akdeniz', 'Karadeniz':'Others', 'İçAnadolu':'Others', 'DoğuAnadolu': 'Others', 'GüneydoğuAnadolu': 'Others'}
data['bölge']=data['bölge'].astype(str).map(mapping2)

data=pd.get_dummies (data, columns= ['bölge'], drop_first=False)

data["booking_value_index"]=data["booking_value_index"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))

mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

data=pd.get_dummies (data, columns= ['booking_value_index'], drop_first=False) 

data=pd.get_dummies (data, columns= ['date_name'], drop_first=False) 

data=data.drop('trivagoId', axis=1)

data['rating'] = data['rating'].fillna(data['rating'].mean)

data = data.sample(frac=1).reset_index(drop=True)

h2o.init()

hf = h2o.H2OFrame(data)
train, valid, test = hf.split_frame([0.6, 0.2], seed=1234)


y = 'clicks'  
x = train.col_names
x.remove(y)


#GBR
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator()
gbm.train(x=x, y=y, training_frame=train, validation_frame=valid)

#gbm.cross_validation_models()
#gbm.cross_validation_metrics_summary()
gbm.varimp_plot()
gbm.varimp()

gbm.mse(train=True, valid=True, xval=False)
gbm.r2(train=True, valid=True, xval=False)
gbm.mae(train=True, valid=True, xval=True)


#RANDOMFOREST
from h2o.estimators.random_forest import H2ORandomForestEstimator
drf = H2ORandomForestEstimator()
drf.train(x=x, y = y, training_frame=train, validation_frame=valid)

drf.varimp_plot()
drf.varimp()

drf.mse(train=True, valid=True, xval=True)
drf.r2(train=True, valid=True, xval=True)
drf.mae(train=True, valid=True, xval=True)


#KERAS

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X=data.drop('clicks', axis=1)
y=data['clicks']

X=X.as_matrix()
y=y.as_matrix()
# define base model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(55, input_dim=55, kernel_initializer='normal', activation='relu'))
    model.add(Dense(28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=20, verbose=0)

kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=20, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


print(r2)

#♥XGB

from sklearn.model_selection import train_test_split
X=data.drop('clicks', axis=1)
y=data['clicks']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.10, random_state=4, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4, shuffle=True) 


xgb=XGBRegressor(learning_rate=0.1, max_depth =7)
xgb.fit(X=X_train, y=y_train)
y_pred=xgb.predict(X_val)
mse_score=mse(y_val,y_pred)
print(mse_score)
r2_score=r2(y_val,y_pred)
print(r2_score)
mae=mae(y_val, y_pred)
print(mae)


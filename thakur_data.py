# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:07:36 2018

@author: tulincakmak
"""

import pandas as pd
import h2o
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import datetime as dt
from datetime import timedelta
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import KFold

data=pd.read_excel('thakur_data_2.xlsx')


print(data['booking_value_index'].unique())
mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

print(data['bolge'].unique())
most_repeated_bolge=data['bolge'].value_counts()
data['bolge']=data['bolge'].fillna(most_repeated_bolge.index[0])


#data['log_date2']=data['log_date']-timedelta(days=2)
data['weekday'] = data[['log_date']].apply(lambda x: dt.datetime.strftime(x['log_date'], '%A'), axis=1)


def miss(x):
    return sum(x.isnull())

missings=data.apply(miss, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(data[i].mean())

data=pd.get_dummies (data, columns= ['bolge'], drop_first=False) 
data=pd.get_dummies(data, columns=['weekday'], drop_first=False)
data=pd.get_dummies(data, columns=['booking_value_index'], drop_first=False)

test_thakur=data.loc[data['log_date']=='2018-05-09']
data2=data.loc[data['log_date']!='2018-05-09']


will_be_dropped={'log_date', 'city_id', 'trivagoId'}
data2=data2.drop(will_be_dropped, axis=1)


h2o.init()

hf = h2o.H2OFrame(data)
#Splitting the data as train, validation, test
train, valid, test = hf.split_frame([0.6, 0.2], seed=1234)
#Dependent & Independent Variables
y = 'clicks'
x = hf.col_names
x.remove(y)

#Gradient Boosting Models
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator(nfolds=3)
gbm.train(x=x, y=y, training_frame=train)
gbm.cross_validation_models()
gbm.cross_validation_metrics_summary()
gbm.varimp_plot()
gbm.varimp()
gbm.mse(train=True, valid=True, xval=True)
gbm.r2(train=True, valid=True, xval=True)
#Random Forest Models
from h2o.estimators.random_forest import H2ORandomForestEstimator
rf = H2ORandomForestEstimator(nfolds=3, keep_cross_validation_predictions=True)
rf.train(x=x, y = y, training_frame=train, validation_frame=valid )
rf.cross_validation_models()
rf.cross_validation_metrics_summary()
rf.varimp_plot()
rf.varimp()
rf.mse(train=True, valid=True, xval=True)
rf.r2(train=True, valid=True, xval=True)
rf.mae(train=True, valid=True, xval=True)

#♦burası hata alıyor bir bak!!!!!!
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
stack = H2OStackedEnsembleEstimator(model_id="my_ensemble", training_frame=train, validation_frame=test, base_models=[gbm.model_id, rf.model_id])
stack.train(x=x, y=y, training_frame=train, validation_frame=valid)
stack.model_performance()



y2=data2['clicks']
X=data2.drop('clicks', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y2, test_size=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

from xgboost import XGBRegressor 
xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb.fit(X=X_train, y=y_train, eval_metric=['rmse'])
y_pred=xgb.predict(X_test)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)
mae_err=mae(y_test,y_pred)   
print(mae_err)    

eval_set = [(X_train, y_train), (X_val, y_val)]
xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["rmse"], eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = xgb.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
r2_score = r2(y_test, predictions)
print("r2_score: %.2f%%" % (r2_score * 100.0))
# retrieve performance metrics
results = xgb.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('Root Mean Squared Error')
pyplot.title('XGBoost Root Mean Squared Error')
pyplot.show()


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(xgb.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

print(importances)
cv = KFold(n_splits=3, shuffle=True)

#train_sizes, train_scores, test_scores = learning_curve(xgb, X, y, n_jobs=-1,cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#test_scores_std = np.std(test_scores, axis=1)
#plt.figure()
#plt.title("XGBRegressor")
#plt.legend(loc="best")
#plt.xlabel("Training examples")
#plt.ylabel("Score")
#plt.gca().invert_yaxis()
#plt.grid()
#plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
#plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
#plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#plt.ylim(-.1,1.1)
#plt.show()
actual=test_thakur['clicks']

will_be_dropped={'log_date', 'city_id', 'trivagoId','clicks'}
test_thakur2=test_thakur.drop(will_be_dropped, axis=1)

y_pred2=xgb.predict(test_thakur2)
y_pred2=pd.DataFrame(data=y_pred2.round())

result=pd.concat([k, y_pred2], axis=1).drop_duplicates().reset_index(drop=True)
trivago_id=pd.DataFrame(data=test_thakur['trivagoId'])

print(r2_score2)

control=pd.read_excel('control.xlsx')
r2_score2 = r2(control['actual'], control['prediction'])
y_pred2.to_csv('prediction_results_0905.csv', index=False)

test_thakur.to_csv('bid.csv', index=False)

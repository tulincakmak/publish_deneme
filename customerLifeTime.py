# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:34:50 2018

@author: tulincakmak
"""
#The r2_score function computes R², the coefficient of determination. 
#It provides a measure of how well future samples are likely to be predicted by the model. 
#Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
#A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats


import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor as ExtraTreeReg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
#from sklearn.tree import ExtraTreeRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet
#from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
#from sklearn.linear_model import BayesianRidge
#from sklearn.linear_model import ARDRegression
#from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import PassiveAggressiveRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.neural_network import MLPRegressor

data=pd.read_excel("CLTV_Vas_data.xlsx")


LABEL="TotalAmount"

data.isnull().sum()
data.describe()
v=data['MobilePrice'].mean()
data['MobilePrice'] = data['MobilePrice'].fillna(v)
opId=pd.get_dummies(data['operatorId'])
data=pd.concat([data,opId], axis=1)
data=data.drop('operatorId', axis=1)
data=data.drop('msisdn', axis=1)

#data.head()
#data=data.drop('offer_id', axis=1)
#data=data.drop('Year', axis=1)
#df=pd.get_dummies(data['Gender'])
#data=pd.concat([data,df], axis=1)
#data=data.drop('Gender', axis=1)
#df_WeekDay=pd.get_dummies(data['Weekday'])
#data=pd.concat([data,df_WeekDay], axis=1)
#data=data.drop('Weekday', axis=1)
#df_month=pd.get_dummies('Month')
#data=pd.concat([data, df_month], axis=1)
#data=data.drop('Month' , axis=1)
#data=data.drop('msisdn', axis=1)



def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label

x_train, y_train, x_test, y_test = split_data(data, 0.20, LABEL)

labels=x_train.columns.values
print(labels)



def RFR(x_train, y_train, x_test, y_test):
    estimator = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    for feature in zip(labels, estimator.feature_importances_):
        print(feature)
    sfm=SelectFromModel(estimator, threshold =0.05)
    sfm.fit(x_train, y_train)
    x_train_important=sfm.transform(x_train)
    x_test_important=sfm.transform(x_test)
    
    estimator_important = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    estimator_important.fit(x_train_important, y_train)
    y_pred_important = estimator_important.predict(x_test_important)
    mse_important=mse(y_test, y_pred_important)
    print("mse_score_important: " + str(mse_important))
    r2_important=r2(y_test, y_pred_important)
    print("r2_score_important: " + str(r2_important))
    
RFR(x_train, y_train, x_test, y_test)
    
def ETR(x_train, y_train, x_test, y_test):
    estimator = ExtraTreeReg(n_estimators=1000, random_state=0, n_jobs=-1)
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    
    
    for feature in zip(labels, estimator.feature_importances_):
        print(feature)
    sfm=SelectFromModel(estimator, threshold=0.05)
    sfm.fit(x_train, y_train)
    x_train_important=sfm.transform(x_train)
    x_test_important=sfm.transform(x_test)
    
    estimator_important = ExtraTreeReg(n_estimators=1000, random_state=0, n_jobs=-1)
    estimator_important.fit(x_train_important, y_train)
    y_pred_important = estimator_important.predict(x_test_important)
    mse_important=mse(y_test, y_pred_important)
    print("mse_score_important: " + str(mse_important))
    r2_important=r2(y_test, y_pred_important)
    print("r2_score_important: " + str(r2_important))

ETR(x_train, y_train, x_test, y_test)

def GBR(x_train, y_train, x_test, y_test):
    estimator = GradientBoostingRegressor(n_estimators=1000, random_state=0)
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    pred=estimator.predict(y_test)
    pred2=pd.concat([y_test, pred], axis=1)
    pred2.to_csv(r"C:\Users\tulincakmak\Desktop\data2.csv", index=False)

    
    
    for feature in zip(labels, estimator.feature_importances_):
        print(feature)
    sfm=SelectFromModel(estimator, threshold=0.05)
    sfm.fit(x_train, y_train)
    x_train_important=sfm.transform(x_train)
    x_test_important=sfm.transform(x_test)
    
    estimator_important = GradientBoostingRegressor(n_estimators=1000, random_state=0)
    estimator_important.fit(x_train_important, y_train)
    y_pred_important = estimator_important.predict(x_test_important)
    mse_important=mse(y_test, y_pred_important)
    print("mse_score_important: " + str(mse_important))
    r2_important=r2(y_test, y_pred_important)
    print("r2_score_important: " + str(r2_important))
    

GBR(x_train, y_train, x_test, y_test)

def AdaBoost(x_train, y_train, x_test, y_test):
    estimator = AdaBoostRegressor(n_estimators=1000, random_state=0)
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    
    
    for feature in zip(labels, estimator.feature_importances_):
        print(feature)
    sfm=SelectFromModel(estimator, threshold=0.05)
    sfm.fit(x_train, y_train)
    x_train_important=sfm.transform(x_train)
    x_test_important=sfm.transform(x_test)
    
    estimator_important = AdaBoostRegressor(n_estimators=1000, random_state=0)
    estimator_important.fit(x_train_important, y_train)
    y_pred_important = estimator_important.predict(x_test_important)
    mse_important=mse(y_test, y_pred_important)
    print("mse_score_important: " + str(mse_important))
    r2_important=r2(y_test, y_pred_important)
    print("r2_score_important: " + str(r2_important))




AdaBoost(x_train, y_train, x_test, y_test)


def Lasso(x_train, y_train, x_test, y_test):
    estimator = LassoLars()
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    

Lasso(x_train, y_train, x_test, y_test)

def Bagging(x_train, y_train, x_test, y_test):
    estimator = BaggingRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    estimator.fit(x_train,y_train)
    t=estimator.score(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    print(t)


Bagging(x_train, y_train, x_test, y_test)



corr = x_train.corr()
plt.figure(figsize=(14,8))
plt.title('Overall Correlation of CC', fontsize=18)
sns.heatmap(corr,annot=False,cmap='BrBG',linewidths=0.2,annot_kws={'size':20})
plt.show()


for feature in labels:
    corr, p_value = pearsonr(x_train[feature], y_train)
    print(feature,corr,p_value)
    
for feature in labels:
    corr, p_value = spearmanr(x_train[feature], y_train)
    print(feature,corr,p_value)
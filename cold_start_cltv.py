# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:57:00 2018

@author: tulincakmak
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR


data=pd.read_excel('cltv_cold_start.xlsx')

data.head()

def missing(x):
    return sum(x.isnull())
    
print (data.apply(missing, axis=0))

data.describe()

data['avgSalePrice'] = data['avgSalePrice'].fillna((data['avgSalePrice'].mean()))
data['avgRentPrice'] = data['avgRentPrice'].fillna((data['avgRentPrice'].mean()))
data['okumamis'] = data['okumamis'].fillna((data['okumamis'].mean()))
data['ilkokul'] = data['ilkokul'].fillna((data['ilkokul'].mean()))
data['ortaokul'] = data['ortaokul'].fillna((data['ortaokul'].mean()))
data['lise'] = data['lise'].fillna((data['lise'].mean()))
data['univ'] = data['univ'].fillna((data['univ'].mean()))
data['evli'] = data['evli'].fillna((data['evli'].mean()))
data['dul'] = data['dul'].fillna((data['dul'].mean()))
data['bosanmis'] = data['bosanmis'].fillna((data['bosanmis'].mean()))
data['bekar'] = data['bekar'].fillna((data['bekar'].mean()))
data['three'] = data['three'].fillna((data['three'].mean()))

city_mapping = {'Istanbul':'Marmara','Kocaeli':'Marmara', 'Manisa':'Ege','Izmir':'Ege','Aydin':'Ege',
                'Denizli':'Ege','Antalya':'Akdeniz','Adana':'Akdeniz','Afyon':'Ege','Bursa':'Marmara','Mersin':'Akdeniz','Isparta':'Akdeniz',
                'Kayseri':'Icand','Eskisehir':'Icand', 'Kütahya':'Ege',
                'Konya':'Icand','Ankara':'Icand','Samsun':'Karadeniz','Gaziantep':'Doguand'}

data['il'] = data['il'].map(city_mapping)
data=pd.get_dummies (data, columns= ['Gender'], drop_first=True)
data=pd.get_dummies (data, columns= ['il'], drop_first=True)

data=data.drop('ilce', axis=1)
data=data.drop('gsm',axis=1)


scaler=MinMaxScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_df, columns=data.columns)
data=scaled_df

#data=data+1
#for i in data.columns:
#    data[i],_=stats.boxcox(data[i])

stats.skew(data['three'])

y=data['three']
X=data.drop(['three'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


nbors=[]
nbors.append(3)
nbors.append(5)
nbors.append(7)
nbors.append(9)
nbors.append(11)
nbors.append(15)
nbors.append(17)
nbors.append(19)
nbors.append(21)
nbors.append(23)
nbors.append(25)

cv_scores=[]

for k in nbors:
    knn=neighbors.KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='mean_squared_error')
    cv_scores.append(scores.mean())
#    knn.fit(X_train,y_train)
#    y_pred=knn.predict(X_test)

MSE = [1 - x for x in cv_scores]
optimal_k = nbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)


plt.plot(nbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('mean_squared_error')
plt.show()

for i in cv_scores:
    print(i)
    
model=SVR(kernel='rbf', gamma='auto', C=1.)
score=cross_val_score(model, X_train, y_train, cv=10, scoring='mean_squared_error')
score.mean()
#model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mse_score=mse(y_test, y_pred)
print(mse_score)
r_score=r2(y_test, y_pred)
print(r_score)
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:51:30 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import scipy.stats as stats

data=pd.read_excel('everything_avarage_28052018.xlsx')

#null kontrolü
def get_missing(x):
    return(sum(x.isnull()))
    
print(data.apply(get_missing))  
  

bolge=data['bolge'].value_counts()
data['bolge']=data['bolge'].fillna(bolge[0])
data['stars']=data['stars'].fillna(3)
data['rating']=data['rating'].fillna(79)
data['hotel_types']=data['hotel_types'].fillna('Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('Summer Hotels', 'Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('City Hotels', 'City Hotel')
data['hotel_types'].unique()

#kategorik kolon kalmadığına emin olduktan sonra çalıştır.!!
missings=data.apply(get_missing, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(0)
    
data=pd.get_dummies(data, columns=['bolge'], drop_first=False)
data=pd.get_dummies(data, columns=['weekday'], drop_first=False)
data=pd.get_dummies(data, columns=['hotel_types'], drop_first=False)

print(data['booking_value_index'].unique())
mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

dropped=['trivagoID', 'log_date']
data=data.drop(dropped, axis=1)

columns_to_numeric=['avg3click_hotelimpr', 'avg7click_hotelimpr', 'avg30click_hotelimpr']

for i in columns_to_numeric:
    data[i]=pd.to_numeric(data[i], errors='ignore')

X=data.drop('class', axis=1)
y=data['class']


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)

columns_to_numeric=['avg3click_hotelimpr', 'avg7click_hotelimpr', 'avg30click_hotelimpr']

for i in columns_to_numeric:
    X_train[i]=pd.to_numeric(X_train[i], errors='ignore')
    
for i in columns_to_numeric:
    X_test[i]=pd.to_numeric(X_test[i], errors='ignore')
    
input_dim=len((X_train.columns))
output_dim=len(y_train.unique())

y=pd.get_dummies(y, drop_first=False )

X=X.as_matrix()
y=y.as_matrix()

from keras.models import Sequential
from keras.layers import Dense

#sigmoid: binary, softmax: multi-class.
model = Sequential()
model.add(Dense(8, input_dim=input_dim, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))
	# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 25, epochs = 100)



y_pred = model.predict_classes(X_test)

y_test2=[]
k=0
l=0
for i in y_test:
    for j in y_test[i]:
        if j>0:
            y_test2.append(k)
        l=l+1
    k=k+1 
    
y_test2=pd.DataFrame(data=y_test2)


ev=model.evaluate(X_train,y_train)

y_pred2=[]

i=0
for  k in y_pred:
    #print(y_pred[i][0])
    t=y_pred[i][0]
    y_pred2.append(t)
    i=i+1

y_pred2=pd.DataFrame(data=y_pred2)
y_test=pd.DataFrame(data=y_test)

y_pred2=y_pred2.round()

y_pred2[0].value_counts()
y_test2[0].value_counts()

print('prediction = %s '% y_pred)
print('accuracy = %s '% ev[1])
print('mean = %s '% ev[0])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


cm=confusion_matrix(y_test2,y_pred2)
print(cm)
f1=f1_score(y_test2,y_pred2, average=None)
print(f1)
cr=classification_report(y_test2,y_pred2)
print(cr)

y_test.to_excel('test_class.xlsx')
y_pred2.to_excel('pred_class.xlsx')



import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred3=model.predict(X_test)
y_pred3=y_pred3.round()

accuracy = accuracy_score(y_test, y_pred3)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

f1=f1_score(y_test, y_pred3, average=None)
print(f1)
cr=classification_report(y_test, y_pred3)
print(cr)

confusion_matrix(y_test,y_pred3)

y_pred3=pd.DataFrame(data=y_pred3)
y_test=pd.DataFrame(data=y_test)

y_pred3[0].value_counts()
y_test['class'].value_counts()




##Bootstrapping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from collections import Counter
#from imblearn.under_sampling import RandomUnderSampler #ImportError: cannot import name 'check_memory'
from imblearn.over_sampling import SMOTE 

print('Original dataset shape {}'.format(Counter(y_train)))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_res)))

class_0=data.loc[data['class']==0]
class_1=data.loc[data['class']==1]
class_2=data.loc[data['class']==2]
class_3=data.loc[data['class']==3]
class_4=data.loc[data['class']==4]
class_5=data.loc[data['class']==5]
class_6=data.loc[data['class']==6]

class_01=pd.concat([class_0, class_1])
class_02=pd.concat([class_0, class_2])
class_03=pd.concat([class_0, class_3])
class_04=pd.concat([class_0, class_4])
class_05=pd.concat([class_0, class_5])
class_06=pd.concat([class_0, class_6])

X_0=class_0.drop('class', axis=1)
y_0=class_0['class']

X_1=class_01.drop('class', axis=1)
y_1=class_01['class']
X_2=class_02.drop('class', axis=1)
y_2=class_02['class']
X_3=class_03.drop('class', axis=1)
y_3=class_03['class']
X_4=class_04.drop('class', axis=1)
y_4=class_04['class']
X_5=class_05.drop('class', axis=1)
y_5=class_05['class']
X_6=class_06.drop('class', axis=1)
y_6=class_06['class']


from sklearn.model_selection import train_test_split
X0_train, X0_val, y0_train, y0_val = train_test_split(X_0, y_0, test_size=0.30)
X1_train, X1_val, y1_train, y1_val = train_test_split(X_1, y_1, test_size=0.30)
X2_train, X2_val, y2_train, y2_val = train_test_split(X_2, y_2, test_size=0.30)
X3_train, X3_val, y3_train, y3_val = train_test_split(X_3, y_3, test_size=0.30)
X4_train, X4_val, y4_train, y4_val = train_test_split(X_4, y_4, test_size=0.30)
X5_train, X5_val, y5_train, y5_val = train_test_split(X_5, y_5, test_size=0.30)
X6_train, X6_val, y6_train, y6_val = train_test_split(X_6, y_6, test_size=0.30)


print('Original dataset shape {}'.format(Counter(y1_train)))
sm = SMOTE(random_state=42)
X1_res, y1_res = sm.fit_sample(X1_train, y1_train)
print('Resampled dataset shape {}'.format(Counter(y1_res)))

print('Original dataset shape {}'.format(Counter(y2_train)))
X2_res, y2_res = sm.fit_sample(X2_train, y2_train)
print('Resampled dataset shape {}'.format(Counter(y2_res)))

print('Original dataset shape {}'.format(Counter(y3_train)))
X3_res, y3_res = sm.fit_sample(X3_train, y3_train)
print('Resampled dataset shape {}'.format(Counter(y3_res)))

print('Original dataset shape {}'.format(Counter(y4_train)))
X4_res, y4_res = sm.fit_sample(X4_train, y4_train)
print('Resampled dataset shape {}'.format(Counter(y4_res)))		

print('Original dataset shape {}'.format(Counter(y5_train)))
X5_res, y5_res = sm.fit_sample(X5_train, y5_train)
print('Resampled dataset shape {}'.format(Counter(y5_res)))  

print('Original dataset shape {}'.format(Counter(y6_train)))
X6_res, y6_res = sm.fit_sample(X6_train, y6_train)
print('Resampled dataset shape {}'.format(Counter(y6_res)))

X0_res=pd.DataFrame(data=X0_train)
X0_res=X0_train.reset_index(drop=True)
X1_res=pd.DataFrame(data=X1_res)
X1_res=X1_res.reset_index(drop=True)
X2_res=pd.DataFrame(data=X2_res)
X2_res=X2_res.reset_index(drop=True)
X3_res=pd.DataFrame(data=X3_res)
X3_res=X3_res.reset_index(drop=True)
X4_res=pd.DataFrame(data=X4_res)
X4_res=X4_res.reset_index(drop=True)
X5_res=pd.DataFrame(data=X5_res)
X5_res=X5_res.reset_index(drop=True)
X6_res=pd.DataFrame(data=X6_res)
X6_res=X6_res.reset_index(drop=True)

y0_res=pd.DataFrame(data=y0_train)
y0_res=y0_train.reset_index(drop=True)
y0_res=y0_res.rename( 'class')
y1_res=pd.DataFrame(data=y1_res)
y1_res=y1_res.reset_index(drop=True)
y1_res=y1_res.rename(columns={0: 'class'})
y2_res=pd.DataFrame(data=y2_res)
y2_res=y2_res.reset_index(drop=True)
y2_res=y2_res.rename(columns={0: 'class'})
y3_res=pd.DataFrame(data=y3_res)
y3_res=y3_res.reset_index(drop=True)
y3_res=y3_res.rename(columns={0: 'class'})
y4_res=pd.DataFrame(data=y4_res)
y4_res=y4_res.reset_index(drop=True)
y4_res=y4_res.rename(columns={0: 'class'})
y5_res=pd.DataFrame(data=y5_res)
y5_res=y5_res.reset_index(drop=True)
y5_res=y5_res.rename(columns={0: 'class'})
y6_res=pd.DataFrame(data=y6_res)
y6_res=y6_res.reset_index(drop=True)
y6_res=y6_res.rename(columns={0: 'class'})


X0=pd.concat([X0_res, y0_res], axis=1)
X1=pd.concat([X1_res,y1_res] ,axis=1)
X2=pd.concat([X2_res,y2_res] ,axis=1)
X3=pd.concat([X3_res,y3_res] ,axis=1)
X4=pd.concat([X4_res,y4_res] ,axis=1)
X5=pd.concat([X5_res,y5_res] ,axis=1)
X6=pd.concat([X6_res,y6_res] ,axis=1)

X1=X1.loc[X1['class']==1]
X2=X2.loc[X2['class']==2]
X3=X3.loc[X3['class']==3]
X4=X4.loc[X4['class']==4]
X5=X5.loc[X5['class']==5]
X6=X6.loc[X6['class']==6]

renamed={'bid': 0 , 'clicks': 1 , 'booking_value_index': 2 , 'stars': 3 , 'rating': 4 , 'avg3clicks': 5 ,
       'avg3click_hotelimpr': 6 , 'avg3beat': 7 , 'avg3meet': 8 , 'avg3lose': 9 ,
       'avg3hotel_impr': 10 , 'avg3bid': 11 , 'avg3opp_cpc': 12 , 'avg3cost': 13 , 'avg3avg_cpc': 14 ,
       'avg3top_pos_share': 15 , 'avg3impr_share': 16 , 'avg3outbidratio': 17 ,
       'avg3unavailability': 18 , 'avg3max_potential': 19 , 'avg7clicks': 20 ,
       'avg7click_hotelimpr': 21 , 'avg7beat': 22 , 'avg7meet': 23 , 'avg7lose': 24 ,
       'avg7hotel_impr': 25 , 'avg7bid': 26 , 'avg7opp_cpc': 27 , 'avg7cost': 28 , 'avg7avg_cpc': 29 ,
       'avg7top_pos_share': 30 , 'avg7impr_share': 31 , 'avg7outbidratio': 32 ,
       'avg7unavailability': 33 , 'avg7max_potential': 34 , 'avg30clicks': 35 ,
       'avg30click_hotelimpr': 36 , 'avg30beat': 37 , 'avg30meet': 38 , 'avg30lose': 39 ,
       'avg30hotel_impr': 40 , 'avg30bid': 41 , 'avg30opp_cpc': 42 , 'avg30cost': 43 ,
       'avg30avg_cpc': 44 , 'avg30top_pos_share': 45 , 'avg30impr_share': 46 ,
       'avg30outbidratio': 47 , 'avg30unavailability': 48 , 'avg30max_potential': 49 ,
       'lastweekbid': 50 , 'lastweekclick': 51 , 'lastdaybid': 52 , 'lastdayclick': 53 , 'avgbid': 54 ,
       'avgclicks': 55 , 'avghotelimpr': 56 , 'holiday_diff': 57 , 'days_of_holiday': 58 ,
       'bolge_Akdeniz': 59 , 'bolge_Doğu Anadolu': 60 , 'bolge_Ege': 61 ,
       'bolge_Güneydoğu Anadolu': 62 , 'bolge_Karadeniz': 63 , 'bolge_Marmara': 64 ,
       'bolge_İç Anadolu': 65 , 'weekday_Friday': 66 , 'weekday_Monday': 67 ,
       'weekday_Saturday': 68 , 'weekday_Sunday': 69 , 'weekday_Thursday': 70 ,
       'weekday_Tuesday': 71 , 'weekday_Wednesday': 72 , 'hotel_types_City ': 73 ,
       'hotel_types_Summer ': 74 , 'hotel_types_Summer Hotel': 75 }

X0=X0.rename(columns=renamed)
    
train=pd.concat([X0, X1, X2, X3, X4, X5, X6])

val_x=pd.concat([X0_val, X1_val, X2_val, X3_val, X4_val, X5_val, X6_val])
val_y=pd.concat([y0_val, y1_val, y2_val, y3_val, y4_val, y5_val, y6_val])

val_x=val_x.rename(columns=renamed)


train_x=train.drop(['class'], axis=1)
train_y=train['class']

model2 = XGBClassifier()
model2.fit(train_x, train_y)

y_pred4=model2.predict(val_x)
y_pred4=y_pred4.round()

accuracy = accuracy_score(val_y, y_pred4)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

f1=f1_score(val_y, y_pred4, average=None)
print(f1)
cr=classification_report(val_y, y_pred4)
print(cr)


confusion_matrix(y_test,y_pred3)

y_pred3=pd.DataFrame(data=y_pred3)
y_test=pd.DataFrame(data=y_test)

y_pred3[0].value_counts()
y_test['class'].value_counts()


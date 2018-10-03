# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:11:35 2018

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:57:51 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train_test_data=[train, test]

#we can get MR,MS,MISS,MASTER from train_test_data into Title column 
for dataset in train_test_data:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.')
    
#pd.crosstab(train['Title'], train['Sex']) 

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    

title_mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
#train.Embarked.unique()
#train.Embarked.value_counts()
    
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


for dataset in train_test_data:
    print(dataset.Embarked.unique())
    dataset['Embarked']=dataset['Embarked'].map({'S': 0, 'C':1, 'Q': 2}).astype(int)
    
    
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

#print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    

train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1
    
#print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
    
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

#familySize ı droplamadan da denenebilir!!!
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)


x_train = train.drop('Survived', axis=1)
y_train = train['Survived']
x_test = test.drop("PassengerId", axis=1).copy()


from keras.models import Sequential
from keras.layers import Dense

input_dim=len(x_train.values[0, :])

x_train=x_train.iloc[:,:].as_matrix()

x_test=x_test.iloc[:,:].as_matrix()

classifier = Sequential()

classifier = Sequential()
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
history=classifier.fit(x_train, y_train, batch_size = 25, epochs = 200)

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


y_pred2=y_pred2.round()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred2.round())


print('accuracy = %s '% ev[1])
print('mean = %s '% ev[0])
print (cm)


print(history.history['val_loss'])

y_pred2.to_csv(r"C:\Users\tulincakmak\Desktop\tt.csv", index=False)


















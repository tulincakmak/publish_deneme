# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:47:20 2018

@author: tulincakmak
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Ridge, Lasso
from sklearn.metrics import  make_scorer,  mean_squared_error
from scipy.stats import skew, skewtest



train_data=pd.read_csv('trainHouse.csv')
test_data=pd.read_csv('testHouse.csv')

ID_train = train_data['Id']
ID_test  = test_data['Id']

train_data.drop("Id", axis = 1, inplace = True)
test_data.drop("Id", axis = 1, inplace = True)

train_data['SalePrice'].describe()

plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

train_data = train_data[train_data['GrLivArea'] < 4500]
# Log transform the target for official scoring
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
y = train_data['SalePrice']

train_data.drop("SalePrice", axis = 1, inplace = True)

print(train_data.shape)
print(test_data.shape)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
print(ntrain)

Combined_data = pd.concat([train_data,test_data]).reset_index(drop=True)
print("Combined size is : {}".format(Combined_data.shape))

Skewed_Features = ['GrLivArea', 'LotArea', 'TotalBsmtSF', '1stFlrSF']

for feature in Skewed_Features:
    
    print((feature), skew(Combined_data[feature]), skewtest(Combined_data[feature]))
    
    Combined_data[feature] = np.log1p(Combined_data[feature])
    
    
total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum()/Combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)


#Combined_data["GarageYrBlt"].value_counts(dropna=False)  this helps look at the distribution

# LotFrontage : NA most likely means no lot frontage
Combined_data["LotFrontage"].fillna(0, inplace=True) 

# Alley : NA most likely means none
Combined_data["Alley"].fillna("None", inplace=True)

# PoolQC : NA means no Pool
Combined_data["PoolQC"].fillna("None", inplace=True)

# MiscFeature : NA means none
Combined_data["MiscFeature"].fillna("None", inplace=True)

# Fence : NA means no fence
Combined_data["Fence"].fillna("None", inplace=True)

# FireplaceQu : NA means no fireplace
Combined_data["FireplaceQu"].fillna("None", inplace=True)

# GarageCond : NA means no garage
Combined_data["GarageCond"].fillna("None", inplace=True)

# GarageFinish : NA means no garage
Combined_data["GarageFinish"].fillna("None", inplace=True)

# GarageQual : NA means no garage
Combined_data["GarageQual"].fillna("None", inplace=True)

# GarageType : NA means no garage
Combined_data["GarageType"].fillna("None", inplace=True)

# GarageType : NA means no garage
Combined_data["GarageYrBlt"].fillna(999999, inplace=True)

# BsmtFinType2 : NA means no basement
Combined_data["BsmtFinType2"].fillna("None", inplace=True)

# BsmtExposure : NA means no basement
Combined_data["BsmtExposure"].fillna("None", inplace=True)

# BsmtQual : NA means no basement
Combined_data["BsmtQual"].fillna("None", inplace=True)

# BsmtFinType1 : NA means no basement
Combined_data["BsmtFinType1"].fillna("None", inplace=True)

# BsmtCond: NA means no basement
Combined_data["BsmtCond"].fillna("None", inplace=True)

# MasVnrType: NA means none
Combined_data["MasVnrType"].fillna("None", inplace=True)

# MasVnrArea : NA most likely means 0
Combined_data["MasVnrArea"].fillna(0, inplace=True) 

# MasVnrArea : NA most likely means 0
Combined_data["Electrical"].fillna("SBrkr", inplace=True) 

# BsmtHalfBath : NA most likely means 0
Combined_data["BsmtHalfBath"].fillna(0, inplace=True)

# BsmtFullBath : NA most likely means 0
Combined_data["BsmtFullBath"].fillna(0, inplace=True)

# BsmtFinSF1 : NA most likely means 0
Combined_data["BsmtFinSF1"].fillna(0, inplace=True)

# BsmtFinSF2 : NA most likely means 0
Combined_data["BsmtFinSF2"].fillna(0, inplace=True)

# BsmtUnfSF : NA most likely means 0
Combined_data["BsmtUnfSF"].fillna(0, inplace=True)

# TotalBsmtSF: NA most likely means 0
Combined_data["TotalBsmtSF"].fillna(0, inplace=True)

# GarageCars : NA most likely means 0
Combined_data["GarageCars"].fillna(0, inplace=True)

# GarageArea : NA most likely means 0
Combined_data["GarageArea"].fillna(0, inplace=True)

# BsmtCond: NA means no basement
Combined_data["Utilities"].fillna(0, inplace=True)

# BsmtCond: NA means no basement
Combined_data["Functional"].fillna(0, inplace=True)

# BsmtCond: NA means no basement
Combined_data["KitchenQual"].fillna(0, inplace=True)

#MSZoning (The general zoning classification) : 'RL' is by far the most common value.

Combined_data["MSZoning"].fillna("RL", inplace=True)

#SaleType : Fill in again with most frequent which is "WD"

Combined_data["SaleType"].fillna("WD", inplace=True)

#Exterior 1 and 2 : Fill in again with most frequent 

Combined_data['Exterior1st'] = Combined_data['Exterior1st'].fillna(Combined_data['Exterior1st'].mode()[0])
Combined_data['Exterior2nd'] = Combined_data['Exterior2nd'].fillna(Combined_data['Exterior2nd'].mode()[0])

Combined_data = Combined_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                  7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                     })

Combined_data['YrSold'] = Combined_data['YrSold'].astype(str)

Combined_data = Combined_data.replace({"Alley" : {"None" : 0, "Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageFinish" : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )

categorical_features = Combined_data.select_dtypes(include = ["object"]).columns
numerical_features = Combined_data.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))

Combined_data_numerical = Combined_data[numerical_features]
Combined_data_categorical = Combined_data[categorical_features]

Combined_data_categorical = pd.get_dummies(Combined_data_categorical,drop_first=True)

Combined_data = pd.concat([Combined_data_categorical, Combined_data_numerical], axis = 1)

print("Combined size is : {}".format(Combined_data.shape))

train_data = Combined_data[:ntrain]
test_data = Combined_data[ntrain:]
test_data = test_data.reset_index(drop=True)

print(train_data.shape)
print(test_data.shape)

X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size = 0.20, random_state = 1)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


scaler = RobustScaler()
X_train.loc[:, numerical_features] = scaler.fit_transform(X_train.loc[:, numerical_features])
X_test.loc[:, numerical_features] = scaler.transform(X_test.loc[:, numerical_features])







import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


input_dim=len(X_train.values[0, :])

X_train=X_train.iloc[:,:].as_matrix()

X_test=X_test.iloc[:,:].as_matrix()



def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    
def wider_model():
	# create model
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.fit(x_train, y_train, batch_size = 25, epochs = 200, validation_data=(x_test, y_test), validation_split=0.2)
    return model

def fit_model(model):
    model.fit(X_train, y_train, batch_size = 25, epochs = 200, validation_data=(X_test, y_test), validation_split=0.2)
    prediction=model.predict(test_data)    #X_test yerine predict ettirilcek dosyayı yaz!!
    return prediction

 
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


scaler_final = RobustScaler()
train_data.loc[:, numerical_features] = scaler_final.fit_transform(train_data.loc[:, numerical_features])

test_data.loc[:, numerical_features] = scaler_final.transform(test_data.loc[:, numerical_features])

test_data=test_data.iloc[:,:].as_matrix()

predictions=fit_model(wider_model())
print(predictions)
prediction=pd.DataFrame(data=predictions)
Final_labels = np.expm1(prediction)
print(Final_labels)

result=pd.DataFrame({'Id': ID_test, 'SalePrice': Final_labels}).to_csv('Predictions2.csv', index =False)  

res=pd.concat([ID_test,Final_labels],axis=1)

res.to_csv('Predictions.csv', index =False)

fit_model(wider_model()) 
fit_model(baseline_model()) 

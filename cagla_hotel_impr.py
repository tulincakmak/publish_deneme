# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:35:21 2018

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:34:07 2018

@author: tulincakmak
"""

from queue import Queue

import pandas
from sqlalchemy import create_engine, MetaData, select, and_ , Table
from sqlalchemy.orm import sessionmaker


class DataResultType:
    raw = "list"
    frame = "frame"
    single = "single"
    void = "void"

    mappers = {
        raw: lambda cursor: [dict(item) for item in cursor.fetchall()],
        void: lambda cursor: None,
        single: lambda cursor: list(cursor.fetchone())[0],
        frame: lambda cursor: pandas.DataFrame(cursor.fetchall(), columns=cursor.keys())
    }


class DatabaseManager:

    def __init__(self, database_name, user, pass_, host):
        """
        Client DB Manager for sql. Manages all sql transactions
        :param database_name:
        """

        self.__db_api_type_codes = {
            1: "string",
            2: "int",
            3: "int",
            4: "datetime",
            5: "float"
        }
        self.__db = database_name
        self.__connection_string = "mssql+pymssql://" + user + ":" + pass_ + "@" + host + "/" + database_name
        self.__engine = create_engine(self.__connection_string, isolation_level="READ COMMITTED", echo=False,
                                      pool_size=50, max_overflow=0)

        """Create main thread connection"""
        self.__connection = self.__engine.connect()

        """Create a list to keep track child thread connections"""
        self.__open_connections = list()
        self.__open_connections.append(self.__connection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
            Close all open connections
        """
        for connection in self.__open_connections:
            if connection:
                try:
                    connection.close()
                except:
                    pass

    def execute_literal(self, statement, result_type="frame"):
        """
            Execute statement
        :param statement: str: literal statement
        :param result_type: DataResultType:
        """

        return self.cursor_to_result(self.__connection.execute(statement), result_type)

    def cursor_to_result(self, cursor, result_type="frame"):
        return DataResultType.mappers[result_type](cursor)

    def bulk_insert_data_frame(self, data_frame, table_name, table=None):
        """
        Inserts a data frame or list of dicts to targeted table
        
        :rtype: None
        """
        table = Table(table_name, MetaData(), autoload=True, autoload_with=self.__engine) if table is None else table
        return self.execute(table.insert(), data_frame)
    
    def get_session(self):
        maker = sessionmaker(bind=self.__engine)
        session = maker(autocommit=True)
        self.__open_connections.append(session)
        return session    
        
    def execute(self, command, data):
        values = data.to_dict(orient="records") if isinstance(data, pandas.DataFrame) else data
        session = self.get_session()
        session.begin(subtransactions=True)
        session.execute(command, values)
        session.commit()

if __name__ == "__main__":
    with DatabaseManager("TempData","tulinC", "tlnckmk", "78.40.231.196") as db:
        data = db.execute_literal("SELECT * from ##dataRoomSpec2")
        
      
        
                                  
                                  

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import scipy.stats as stats
#import pyodbc
#data=pd.read_excel('everything_avarage_28052018.xlsx')





#cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
#cursor = cnxn.cursor()
#
#sql="select * from [otelz__1521458750].[dbo].[pred_data_click]"
#data = pd.read_sql(sql,cnxn)

#sql="select * from ##deneme3"
#r = pd.read_sql(sql,cnxn)

#data['CTR']=data['clicks']/data['hotel_impr']

#delete=['clicks','my_min_position', 'my_price', 'top4_min_price','total_min_price', 'hotels_position',  'hotels_price', 'odamax_position','odamax_price']

delete=['clicks','click_hotel_impr', 'Id']

data=data.drop(delete, axis=1)


#null kontrolü
def get_missing(x):
    return(sum(x.isnull()))
    
#print(data.apply(get_missing))  
#data=data.drop('region', axis=1)

#bolge=data['bolge'].value_counts()
#data['bolge']=data['bolge'].fillna(bolge[0])
#data['stars']=data['stars'].fillna(3)
#data['rating']=data['rating'].fillna(79)
#data['hotel_types']=data['hotel_types'].fillna('Summer Hotel')

#data['hotel_types']=data['hotel_types'].replace('Summer Hotels', 'Summer Hotel')
#data['hotel_types']=data['hotel_types'].replace('City Hotels', 'City Hotel')
data['hotel_types']=data['hotel_types'].replace('Summer ', 'Summer')
#data['hotel_types'].unique()

#kategorik kolon kalmadığına emin olduktan sonra çalıştır.!!
#data['weather'].unique()
#mapp={'azbulutlu': 'bulutlu', 
#'bulutlu': 'bulutlu',
#'cogunluklabulutlu': 'bulutlu',
#'cogunluklabulutlusaganakyagisli': 'yagmurlu',
#'gokgurultulusagnakyagmur': 'yagmurlu',
#'gunesli': 'gunesli',
#'parcalibulutlu': 'bulutlu',
#'parcaliguneslisaganakyagis': 'yagmurlu',
#'saganakyagis': 'yagmurlu',
#'sisli': 'sisli',
#'yagmurlu': 'yagmurlu',
#'yogunbulutlu': 'bulutlu'}  
#data['weather']=data['weather'].map(mapp)
data=pd.get_dummies(data, columns=['bolge'], drop_first=False)
data=pd.get_dummies(data, columns=['weekday'], drop_first=False)
data=pd.get_dummies(data, columns=['hotel_types'], drop_first=False)
data=pd.get_dummies(data, columns=['weather'], drop_first=False)

data=data.drop('Status', axis=1)

#print(data['booking_value_index'].unique())
#mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
#data['booking_value_index'].unique()
#data['booking_value_index']=data['booking_value_index'].map(mapping)

#data=data.drop('Weather_Status', axis=1)

data=data.drop('my_roomSpec',axis=1)  
data=data.drop('odamax_roomSpec',axis=1)  
data=data.drop('hotels_roomSpec',axis=1)  

data['my_Class']=data['my_Class'].fillna('standart')
data['hotels_Class']=data['hotels_Class'].fillna('standart')
data['odamax_Class']=data['odamax_Class'].fillna('standart')

data=pd.get_dummies(data, columns=['my_Class'], drop_first=False)
data=pd.get_dummies(data, columns=['hotels_Class'], drop_first=False)
data=pd.get_dummies(data, columns=['odamax_Class'], drop_first=False)

missings=data.apply(get_missing, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(0)

columns_to_numeric=['avg3click_hotelimpr', 'avg7click_hotelimpr', 'avg30click_hotelimpr']

for i in columns_to_numeric:
    data[i]=pd.to_numeric(data[i], errors='ignore')
  

object_cols=data.dtypes
convert_float=[]

for i in object_cols.index:
    if object_cols[i]== 'object' and i!='log_date':
        convert_float.append(i)
        
        
for i in convert_float:
    data[i]=pd.to_numeric(data[i], errors='ignore')

data2=data

missings=data.apply(get_missing, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(0)
    
test2=data.loc[data['log_date']=='2018-09-11']
train2=data.loc[data['log_date']<'2018-09-11'] 

#data=data.drop(['trivago_id','log_date'], axis=1) 



train=train2.drop(['trivago_id','log_date'], axis=1) 
test=test2.drop(['trivago_id','log_date'], axis=1) 

missings=train.apply(get_missing, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    train[i]=train[i].fillna(0)
    
missings=test.apply(get_missing, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    test[i]=test[i].fillna(0)

X=train.drop('hotel_impr', axis=1)
y=train['hotel_impr']
label=test['hotel_impr']
test=test.drop('hotel_impr', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)    

from sklearn.preprocessing import StandardScaler    
scaler = StandardScaler()
scaler.fit(X_train)   

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test=scaler.transform(test)

from sklearn.decomposition import PCA
pca = PCA(.95)

import time

t0 = time.time()
pca.fit(X_train) 
t1 = time.time()

total = t1-t0

print(total)   

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
test=pca.transform(test)


#pca.score(X)

#train=train.sample(frac=1).reset_index(drop=True)
#test=test.sample(frac=1).reset_index(drop=True)

#test_big=test_big.sample(frac=1).reset_index(drop=True)


#train2=train.drop(['trivago_id'], axis=1)
#train2=train2.drop(['log_date'], axis=1)
#
#test1=test.drop(['trivago_id'], axis=1)
#test1=test1.drop(['log_date'], axis=1)

#test_big1=test_big.drop(['trivago_id'], axis=1)

#test_big1=test_big1.drop(['log_date'], axis=1)

test1=pd.DataFrame(data=test)
train2=pd.DataFrame(data=train)



#data['my_Class'].dtype




from xgboost import XGBRegressor 
xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)

t2 = time.time()
xgb.fit(X=X_train, y=y_train, eval_metric=['rmse'])
t3 = time.time()

total2 = t3-t2

print(total2)  



y_pred=xgb.predict(test)

r2_score=r2(label,y_pred)
print(r2_score)
mse_score=mse(label, y_pred)
print(mse_score)
mae_score=mae(label, y_pred)
print(mae_score)

#importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(xgb.feature_importances_,3)})
#importances = importances.sort_values('importance',ascending=False).reset_index()
#print(importances)
#
#importances.to_excel('ctr_importance_0409.xlsx')
#
#actual_y=test1['hotel_impr']
#test2=test1.drop('hotel_impr', axis=1)

#test_big2=test_big1.drop('hotel_impr', axis=1)


#from xgboost import plot_importance
##h = 30
##w = 30
##fig, ax = plt.subplots(figsize=(h, w))
##plot_importance(xgb00,importance_type = 'cover',ax=ax)
#import matplotlib.pyplot as plt
#import seaborn as sns
#h = 50
#w = 50
#fig, ax = plt.subplots(figsize=(h, w))
#plot_importance(xgb,importance_type = 'cover',ax=ax)



#mae_err=mae(y_val,y_pred)   
#print(mae_err)  



y_pred1=xgb.predict(test)    
y_pred2=pd.DataFrame(data=y_pred)  
#test=test.reset_index()


#y_pred_big1=xgb.predict(test_big2)    
#y_pred_big2=pd.DataFrame(data=y_pred_big1)  
#test_big=test_big.reset_index()

label=label.reset_index()
y_pred2=y_pred2.reset_index()
result=pd.concat([y_pred2,  label], axis=1)
result.to_excel('impr_pred_pca.xlsx')


#result_big=pd.concat([y_pred_big2, test_big['trivago_id'], test_big['hotel_impr']], axis=1)
#result_big.to_excel('impr_pred_for3007_big.xlsx')

#test_bigger['trivago_id'].to_excel('bigger.xlsx')



#calc=pd.read_excel('ctr_pred_for0409.xlsx')
#
#r2_0807=r2(calc['click'], calc['click_pred'])
#print(r2_0807)
#
#
#import seaborn as sns
#sns.distplot(y_pred2 ,hist = False, color = 'black')
#sns.distplot(actual_y, hist = False, color = 'r')
#
#y_val=y_val.reset_index()
#y_pred=pd.DataFrame(data=y_pred)
#merged=pd.concat([y_pred, y_val], axis=1)
#merged=merged.drop([ 'index'], axis=1)
#
#merged=merged.drop([ 'level_0'], axis=1)
#import seaborn as sns
#sns.set(style="ticks")
#
## Load the example dataset for Anscombe's quartet
#merged=merged.rename(columns={0: "predicted", "hotel_impr": "actual"})
#
#merged2=merged.rename(columns={0: "predicted"})
#
#merged2['predicted']=merged2['predicted'].replace(merged2['predicted'].loc[merged2['predicted']<0], 0)
#merged2['predicted']=merged2['predicted'].round()
#
## Show the results of a linear regression within each dataset
#sns.lmplot(x="actual", y="predicted", data=merged2)
#
#





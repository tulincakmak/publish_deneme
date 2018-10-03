# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:52:56 2017

@author: tulincakmak
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

tf.logging.set_verbosity(tf.logging.INFO)


LABEL = 'Exited'


pred=[]


train_data = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True,
                              header=0)
test_set = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True,
                          header=0)
  
test_label = test_set[LABEL].astype(float)
test_set.drop("Surname", axis = 1, inplace=True)
test_set.drop("RowNumber", axis = 1, inplace=True)
test_set.drop("CustomerId", axis = 1, inplace=True)
train_data.drop("CustomerId", axis = 1, inplace=True)
train_data.drop("Surname", axis = 1, inplace=True)
train_data.drop("RowNumber", axis = 1, inplace=True)

df2=train_data.select_dtypes(include=['object'])
drop_cols=df2.columns
train_data.drop(drop_cols, axis = 1, inplace=True)

df=test_set.select_dtypes(include=['object'])
drop_cols=df.columns
test_set.drop(drop_cols, axis = 1, inplace=True)


def get_input_fn(data_set, num_epochs=None, shuffle=False):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in data_set.columns}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


feature_cols = [tf.feature_column.numeric_column(k) for k in train_data.columns]

#kAYDETME İŞLEMİ BAŞLIYOR!!
x=tf.Variable(['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary'], name='x')
y=tf.Variable(['Exited'], name='y')


init_op = tf.initialize_all_variables()
saver=tf.train.Saver([x,y])
with tf.Session() as sess:  
    sess.run(init_op)
    regressor = tf.estimator.LinearRegressor(feature_columns=feature_cols)

    regressor.train(input_fn=get_input_fn(train_data), steps=500)
    ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    saver.save(sess, 'tmprtcqtf7o\model.ckpt')  #KAYDEDİLEN DOSYANIN DEĞİŞMESİ LAZIM!!

#bU KISIMDA BİTTİ
y = regressor.predict(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))

predictions = list(y)
 
for prediction in predictions:
    print(prediction['predictions'][0])
    t=prediction['predictions'][0]
    pred.append(t)
         


#bu fonksiyon dışarıdan tüm algoritmalar için çağırılacak.
def calculate(prediction, LABEL):
    arr = {"accuracy": accuracy_score(prediction , LABEL),
           "report": classification_report(prediction , LABEL),
           "Confusion_Matrix": confusion_matrix(prediction , LABEL),
           "F1 score" : f1_score(prediction , test_label),
           "Recall Score": recall_score(prediction , LABEL),
           "cohen_kappa": cohen_kappa_score(prediction , LABEL)
           }
    return arr
 
    
pred1=pd.DataFrame(data=pred)


print(calculate(pred1.round(),test_label))






# -*- coding: utf-8 -*-
"""@author: tulincakmak"""

#logistic regression

import pandas as pd
import tensorflow as tf
import numpy as np
import tempfile


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


COLS=["RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age","Tenure",
      "Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]


FEATURES = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
           "HasCrCard","IsActiveMember", "EstimatedSalary"]

LABEL="Exited"

df_train = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True, header=0)
df_test = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True, header=0)

test_label = df_test[LABEL].astype(float)
df_test.drop("Surname", axis = 1, inplace=True)
df_test.drop("RowNumber", axis = 1, inplace=True)
df_test.drop("CustomerId", axis = 1, inplace=True)
df_train.drop("CustomerId", axis = 1, inplace=True)
df_train.drop("Surname", axis = 1, inplace=True)
df_train.drop("RowNumber", axis = 1, inplace=True)
df_train.drop("Geography", axis = 1, inplace=True)
df_train.drop("Gender", axis = 1, inplace=True)
#Bu fonksiyonda test ve label olarak ayırıyor????
def get_input_fn():
  return {'x': tf.constant(df_train[FEATURES].as_matrix(), tf.float32, df_train.shape),
          'y': tf.constant(df_train[LABEL].as_matrix(), tf.float32, df_train.shape)
          }
          

print(df_train)

df=df_train.select_dtypes(exclude=['object'])
numeric_cols=df.columns
print(numeric_cols)


    
df2=df_train.select_dtypes(include=['object'])
categorical_cols=df2.columns



all_cols=df_train.columns    
model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=[numeric_cols])


#train data
m.train(input_fn=get_input_fn ,steps=5000)

#you can get accuracy, accuracy_baseline, auc, auc_precision_recall, average_loss, global_step, label/mean, lossprediction/mean
results = m.evaluate(input_fn= get_input_fn(df_test, num_epochs=1, shuffle=False),steps=None)


print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))

#get prediction results
y = m.predict(input_fn=get_input_fn(df_test, num_epochs=1, shuffle=False))
pred = list(y)
#print(list(y))

#sonuçları bastır!!
rowNumber=0
for i in pred:
    print(str(rowNumber)+': '+str(pred[i]))
    rowNumber=rowNumber+1
    
  # AVOID OVERFITTING 
#m = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    #optimizer=tf.train.FtrlOptimizer(
     #learning_rate=0.1,
      #l1_regularization_strength=1.0,
      #l2_regularization_strength=1.0),
    #model_dir=model_dir)

    
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


    
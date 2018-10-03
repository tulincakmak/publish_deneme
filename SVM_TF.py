# -*- coding: utf-8 -*-


import pandas as pd
import tensorflow as tf
import numpy as np
import tempfile
from sklearn.model_selection import train_test_split

def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label

FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary"]

LABEL="Exited"


data = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True, header=0)

data.drop("Surname", axis=1, inplace=True)
#data.drop("RowNumber", axis=1, inplace=True)
data.drop("CustomerId", axis=1, inplace=True)
data.drop("Geography", axis=1, inplace=True)
data.drop("Gender", axis=1, inplace=True)
    
x_train, y_train, x_test, y_test = split_data(data, 0.20, LABEL)
    
    

def get_input_fn_train():
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=x_train.astype('float64'),
            y=y_train,
            shuffle=False
        )
        return input_fn

def get_input_fn_test():
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=x_test,
            y=y_test,
            shuffle=False,
        )
        return input_fn
 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(get_input_fn_train()) 

   
example_id= tf.as_string(x_train['RowNumber'])

#base_columns=[gender,Geography,CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]

model_dir = tempfile.mkdtemp()

estimator = tf.contrib.learn.SVM(example_id_column= example_id,
                                 feature_columns=feature_columns, l2_regularization=10.0, model_dir=model_dir)


estimator.fit(input_fn=get_input_fn_train(), steps=100)


results=estimator.evaluate(input_fn=get_input_fn_test(), steps=1)


for key in sorted(results):
  print("%s: %s" % (key, results[key]))


pred=list(estimator.predict(input_fn=get_input_fn_test()))


tensor=tf.convert_to_tensor(y_train)
tensor.get_shape()


print(str(x_train['RowNumber']))


print(example_id)
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:22:37 2017

@author: tulincakmak
"""

import pandas as pd
import tensorflow as tf
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label




    # logistic regression

COLS = ["RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]

FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary"]

LABEL = "Exited"

data = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True, header=0)

data.drop("Surname", axis=1, inplace=True)
data.drop("RowNumber", axis=1, inplace=True)
data.drop("CustomerId", axis=1, inplace=True)
data.drop("Geography", axis=1, inplace=True)
data.drop("Gender", axis=1, inplace=True)
x_train, y_train, x_test, y_test = split_data(data, 0.20, LABEL)


# Bu fonksiyonda test ve label olarak ayırıyor????
def get_input_fn_train():
    input_fn = tf.estimator.inputs.pandas_input_fn(
        x=x_train,
        y=y_train,
        shuffle=False
    )
    return input_fn

def get_input_fn_test():
    input_fn = tf.estimator.inputs.pandas_input_fn(
        x=x_test,
        y=y_test,
        shuffle=False
    )
    return input_fn


x_train = x_train.select_dtypes(exclude=['object'])
numeric_cols = x_train.columns

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(get_input_fn_train())

#VERİLERİ BU NOKTADA KAYDETMEYE BAŞLITOR!!!
x=tf.Variable(['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary'], name='x')
y=tf.Variable(['Exited'], name='y')


init_op = tf.initialize_all_variables()
saver=tf.train.Saver([x,y])
with tf.Session() as sess:  
    sess.run(init_op)
    model_dir = tempfile.mkdtemp()
    m = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=feature_columns)

    # train data
    m.train(input_fn=get_input_fn_train(), steps=5000)

    # you can get accuracy, accuracy_baseline, auc, auc_precision_recall, average_loss, global_step, label/mean, lossprediction/mean
    results = m.evaluate(input_fn=get_input_fn_test(), steps=None)

    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    
    saver.save(sess, 'tmprtcqtf7o\model.ckpt')

# get prediction results
y = m.predict(input_fn=get_input_fn_test())
predictions = list(y)
  
pred=pd.DataFrame(data=predictions)
pred2=pd.DataFrame(data=pred['class_ids'])


pred3=[]

rowNumber=0
for row in pred2["class_ids"]:
    pred3.append(row[0])
    print(str(rowNumber) + ': ' + str(row[0]))
    rowNumber = rowNumber + 1


    # AVOID OVERFITTING
    # m = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    # optimizer=tf.train.FtrlOptimizer(
    # learning_rate=0.1,
    # l1_regularization_strength=1.0,
    # l2_regularization_strength=1.0),
    # model_dir=model_dir)


def calculate(prediction, LABEL):
    arr = {"accuracy": accuracy_score(prediction, LABEL),
           "report": classification_report(prediction, LABEL),
           "Confusion_Matrix": confusion_matrix(prediction, LABEL),
           "F1 score": f1_score(prediction, LABEL),
           "Recall Score": recall_score(prediction, LABEL),
           "cohen_kappa": cohen_kappa_score(prediction, LABEL)
           }
    return arr

print(calculate(pred3, y_test))


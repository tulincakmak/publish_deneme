# -*- coding: utf-8 -*-


import  tensorflow as tf
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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



#if __name__ == "__main__":


LABEL = "Exited"

data = pd.read_csv("Churn_Modelling.csv", skipinitialspace=True, header=0)

data.drop("Surname", axis=1, inplace=True)
data.drop("RowNumber", axis=1, inplace=True)
data.drop("CustomerId", axis=1, inplace=True)
data.drop("Geography", axis=1, inplace=True)
data.drop("Gender", axis=1, inplace=True)


x_train, y_train, x_test, y_test = split_data(data, 0.20, LABEL)
    
    

def get_input_fn_train():
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=x_train.astype('float64'),
            y=y_train.astype('float32'),
            shuffle=False
        )
        return input_fn

def get_input_fn_test():
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=x_test.astype('float64'),
            y=y_test.astype('float32'),
            shuffle=False
        )
        return input_fn

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
    estimator= tf.contrib.learn.LinearEstimator(model_dir=model_dir, feature_columns=feature_columns,
                                                head=tf.contrib.learn.poisson_regression_head())

    estimator.fit(input_fn=get_input_fn_train(),steps=5000)


    results=estimator.evaluate(input_fn=get_input_fn_test())

    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    
    saver.save(sess, 'tmprtcqtf7o\model.ckpt')
    
   
 
#BU NOKTADA DA BİTİYOR.
  
  
pred=list(estimator.predict(input_fn=get_input_fn_test()))

pred1=pd.DataFrame(data=pred)


rowNumber = 0
for i in pred1['scores']:
    print(str(rowNumber) + ': ' + str(i))   #PREDİCTİON RESULTS
    rowNumber = rowNumber + 1
    
test=pd.DataFrame(data=y_test)

acc=accuracy_score(pred1 , test)
cr=classification_report(pred1 , test)
f1=f1_score(pred1 , test)
cm=confusion_matrix(pred1 , test)
ccs=cohen_kappa_score(pred1 , test)

print('accuracy = %s'% acc)
print('classification report = %s'% cr)
print('F1 score = %s'% f1)
print('Confusion matrix = %s'% cm)
print('cohen cappa = %s'% ccs)


print(pred1)

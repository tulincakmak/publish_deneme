# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:36:10 2018

@author: tulincakmak
"""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

twenty_train.target_names #prints all the categories
print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file

controlled.head()

controlled2=controlled.drop('Id', axis=1)
controlled3=controlled2['RoomSpec'].tolist()
train_label=controlled2['Class'].tolist()

will_be_checked2=will_be_checked.drop('Id', axis=1)
will_be_checked3=will_be_checked2['RoomSpec'].tolist()
test_label=will_be_checked2['Class'].tolist()

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(controlled3)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
X_train_tfidf.dtype

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, controlled2['Class'])

#burada kaldım

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
text_clf = text_clf.fit(controlled3, controlled2['Class'])


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(controlled3, controlled2['Class'])
predicted2 = text_clf_svm.predict(will_be_checked3)

#import numpy as np
#twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted2 = text_clf_svm.predict(will_be_checked3)
predicted2=pd.DataFrame(data=predicted2)
predicted2=predicted2.reset_index()
will_be_checked=will_be_checked.reset_index()
result=pd.concat([will_be_checked, predicted2], axis=1)
result.to_excel('BoW_results.xlsx')
#np.mean(predicted == will_be_checked2['Class'])

from sqlalchemy import create_engine
engine = create_engine('mssql+pymssql://gizemaras:gzmrs@123@78.40.231.196/TempData', echo=False) 
result.to_sql('text_classification_12092018', con=engine, if_exists='append',chunksize=1000)
engine.execute("SELECT * FROM levensthein").fetchall()


################################################################################################################################################



X=controlled['RoomSpec']
y=controlled['Class']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

X_train2=X_train.tolist()
y_train2=y_train.tolist()

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train2)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
#X_train_tfidf.dtype

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)



from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
text_clf = text_clf.fit(X_train, y_train)

#import numpy as np
#twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(X_val)
np.mean(predicted == y_val)


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_val)
np.mean(predicted_svm == y_val)

from sklearn.model_selection import GridSearchCV

#MultinomialNB için Grid Search
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)

gs_clf.best_score_
gs_clf.best_params_


#SVM için GridSearch
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
gs_clf_svm.best_score_
gs_clf_svm.best_params_
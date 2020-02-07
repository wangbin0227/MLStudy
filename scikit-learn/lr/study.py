### -*- encoding:utf-8 -*-

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('./SMSSpamCollection', delimiter='\t', header=None)
print (df.head())

spam_num = df[df[0] == 'spam'][0].count()

ham_num = df[df[0] == 'ham'][0].count()
print (ham_num)


X = df[1].values
y = df[0].values
print (X)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)


predictions = classifier.predict_proba(X_test)
print (predictions)
print (predictions[:, 1])


from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print (scores)

from sklearn.metrics import precision_score
precisions = precision_score(classifier, X_train, y_train, cv=5)
print (precisions)

recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print (recalls)

### -*- encoding:utf8 -*-

from sklearn.datasets import fetch_20newsgroups
categories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']

newsgroups_train = fetch_20newsgroups(
        subset='train', 
        categories=categories,
        remove=('headers', 'footers', 'quotes')
        )

newsgroups_test = fetch_20newsgroups(
        subset='test', 
        categories=categories,
        remove=('headers', 'footers', 'quotes')
        )

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

from sklearn.linear_model import Perceptron
clf = Perceptron(random_state=11)
clf.fit(X_train, newsgroups_train.target)

from sklearn.metrics import classification_report
predictios = clf.predict(X_test)
print (classification_report(newsgroups_test.target, predictios))



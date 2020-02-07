## -*- encoding:utf8 -*-

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

data = [
    "abc abc abc",
    'abc abd'
    ]

res = vectorizer.fit_transform(data).todense()
print (res)
print (vectorizer.vocabulary_)

from sklearn.metrics.pairwise import euclidean_distances

print (euclidean_distances(res[0], res[1]))

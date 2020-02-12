### -*- encoding:utf8 -*-


import numpy as np

c1x = np.random.uniform(0.5, 1.5, (1, 10))
c1y = np.random.uniform(0.5, 1.5, (1, 10))
print (c1x)

c2x = np.random.uniform(3.5, 4.5, (1, 10))
c2y = np.random.uniform(3.5, 4.5, (1, 10))

x = np.hstack((c1x, c2x))
y = np.hstack((c1y, c2y))
print (x)
print (y)

X = np.vstack((x, y)).T
print (X)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

K = range(1, 10)
meanDispersions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meanDispersions.append(sum(np.min(cdist(X,kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
print (meanDispersions)

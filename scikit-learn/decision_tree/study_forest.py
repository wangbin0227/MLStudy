## -*- encoding:utf8 -*- 

from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples = 1000,
    n_features = 100,
    n_informative=20,
    n_clusters_per_class = 2,
    random_state=1
    )
print (X)
print (len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=11)

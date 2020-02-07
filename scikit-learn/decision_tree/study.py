## -*- encoding:utf8 -*-


### 读取数据
import pandas as pd
df = pd.read_csv('./data/ad.data', header=None)


### 样本生成
from sklearn.model_selection import train_test_split
print (df.columns.values)
explanatory_variable_columns = set(df.columns.values)
explanatory_variable_columns.remove(len(df.columns.values) - 1)
response_variable_column = df[len(df.columns.values) -1]
y = [1 if _ == 'ad.' else 0 for _ in response_variable_column]

X = df[list(explanatory_variable_columns)].copy()
X.replace(to_replace=' *?', value=-1, regex=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)


### 模型构建
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
    ])

parameters = {
        'clf__max_depth': (150, 155, 160),
        'clf__min_samples_split': (2, 3),
        'clf__min_samples_leaf': (1, 2, 3),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)

from sklearn.metrics import classification_report
predictions = grid_search.predict(X_test)
print (classification_report(y_test, predictions))


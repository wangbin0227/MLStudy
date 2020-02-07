### -*- encoding:utf-8 -*-

## 参数调优
## 使用网格搜索

### 读取数据
import pandas as pd
df = pd.read_csv('./SMSSpamCollection', delimiter='\t', header=None)
X = df[1].values
y = df[0].values

### 数据预处理
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

### 模型构建
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
    ])

parameters = {
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__stop_words': ('english', None),
        'vect__max_features': (2500, 5000, 10000, None),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__penalty': ('l2', 'l1'),
        'clf__C': (0.01, 0.1, 1, 10),
    }

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

print (grid_search.best_score_)


best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print (param_name, best_parameters[param_name])

from sklearn.metrics import precision_score, recall_score
predictions = grid_search.predict(X_test)
print (precision_score(y_test, predictions))
print (recall_score(y_test, predictions))


# coding: utf-8

# ## Grow a decision tree, each node is a "cluster" 
# 

# In[1]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn import cross_validation, grid_search
from sklearn.cross_validation import cross_val_score 
from modeling_funcs import *
from IPython.display import display
import seaborn as sns

sns.set(color_codes=True)
plt.rcParams['figure.figsize'] = (10.0, 10.0)


# In[2]:

vectorized_data = pd.read_csv('../all_data_vectorized.csv', sep='|', index_col=0)
slope = pd.read_csv('../all_slope.csv', sep = '|', index_col=0)
all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )

everybody = vectorized_data.join(slope)


# In[3]:

q = np.percentile(everybody.ALSFRS_slope, range(0,100,10))
q


# In[4]:

everybody.loc[:, 'ALSFRS_bin'] = np.digitize(everybody.ALSFRS_slope, q)
display(everybody[['ALSFRS_slope','ALSFRS_bin']].describe())


# In[5]:

sns.distplot(everybody.ALSFRS_slope, rug=True, kde=False);


# In[39]:

from sklearn import tree

X = everybody.drop(['ALSFRS_slope','ALSFRS_bin'], 1)
y = everybody.ALSFRS_slope

clf = grid_search.GridSearchCV(tree.DecisionTreeRegressor(min_samples_split = 275), 
                               {'min_samples_leaf': range(10,150,10)})
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[40]:

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

def pred_vs_actual(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    clf.fit(X_train, y_train)
    print clf.best_estimator_
    print "train", mean_squared_error(y_train, clf.predict(X_train))    
    res = X_test.apply(lambda s: pd.Series({'prediction': clf.predict(s)[0]}) , axis=1)
    res.loc[:, 'actual'] = y_test
    res.loc[:, 'SE'] = res.apply(lambda s: (s['prediction'] - s['actual'])**2 , axis=1)
    print "test", mean_squared_error(res.actual, res.prediction), np.mean(res.SE)
    display(res.sort(['SE']).head(3))
    display(res.sort(['SE']).tail(15))
    return res
    
res = pred_vs_actual(clf, X, y)
sns.distplot(res.SE, rug=True);


# In[41]:

from sklearn.externals.six import StringIO
with open("../tree.dot", 'w') as f:
    f = tree.export_graphviz(clf.best_estimator_, feature_names = X.columns, out_file=f)


# In[42]:

res.describe()


# In[67]:

clf = linear_model.LinearRegression()
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[152]:

clf = grid_search.GridSearchCV(linear_model.Ridge(), {'alpha': np.arange(45,145,0.1)})
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[153]:



display(pred_vs_actual(clf, X, y).head())


# In[157]:

clf = grid_search.GridSearchCV(linear_model.Lasso(), {'alpha': np.arange(0.01,0.21,0.01)})
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()
display(pred_vs_actual(clf, X, y).head())


# In[158]:

print [(X.columns[i[0]], i[1]) for i in sorted(enumerate(clf.best_estimator_.coef_), key=lambda x:-abs(x[1])) if i[1] <> 0]


# In[159]:

from sklearn.ensemble import RandomForestClassifier
clf = grid_search.GridSearchCV(RandomForestClassifier(random_state=0), 
                               {'min_samples_split': range(5,15,5), 'n_estimators': [1000]})
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()
display(pred_vs_actual(clf, X, y).head())


# In[164]:

clf.best_estimator_.feature_importances_

print [(X.columns[i[0]], i[1]) for i in sorted(enumerate(clf.best_estimator_.feature_importances_), key=lambda x:-abs(x[1]))]


# In[71]:

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=1000, min_samples_split=10, random_state=0)
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[72]:

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=1000)
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[73]:

from sklearn import svm
clf = svm.SVC(C = 1, gamma = 0.01, kernel = 'rbf')
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[74]:

from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(svm.LinearSVC(), max_samples=0.5, max_features=0.5)
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[79]:

from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# In[76]:

from sklearn import grid_search
parameters = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
 ]
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
print cross_val_score(clf, X, y, scoring="mean_squared_error").mean()


# ## Pickle the clustering model

# In[77]:

clustering_model = {"model": clf}
pickle.dump( clustering_model, open('../tree_model.pickle', 'wb') )


# In[78]:


for t in ['all', 'test']:
    cur_data = pd.read_csv('../' + t + '_data_vectorized.csv', sep = '|', error_bad_lines=False, index_col="SubjectID")
    cur_data = cur_data[clustering_columns]
    res = pd.DataFrame(index = cur_data.index.astype(str)) # SubjectID is always str for later joins
    res['cluster'] = tree.predict(cur_data)
    print np.bincount(res.cluster)
    print t, res.shape
    res.to_csv('../' + t + '_tree_clusters.csv',sep='|')


# In[ ]:

res.head()


# In[ ]:




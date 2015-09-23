
# coding: utf-8

# ## The best we can do with trees
# This is the upper boundary of what we can achieve with trees, using all features
# 

# In[48]:

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


# In[49]:

vectorized_data = pd.read_csv('../all_data_vectorized.csv', sep='|', index_col=0)
slope = pd.read_csv('../all_slope.csv', sep = '|', index_col=0)
all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )

everybody = vectorized_data.join(slope)


# In[50]:

q = np.percentile(everybody.ALSFRS_slope, range(0,100,10))
everybody.loc[:, 'ALSFRS_bin'] = np.digitize(everybody.ALSFRS_slope, q)
display(everybody[['ALSFRS_slope','ALSFRS_bin']].describe())


# In[51]:

sns.distplot(everybody.ALSFRS_slope, rug=True, kde=False);


# In[52]:

from sklearn import tree

X = everybody.drop(['ALSFRS_slope','ALSFRS_bin'], 1)
y = everybody.ALSFRS_slope

clf = grid_search.GridSearchCV(tree.DecisionTreeRegressor(), 
                               {'min_samples_split':range(250,270,5), 'min_samples_leaf': range(50,70,5)})
print np.sqrt(-1.0 * cross_val_score(clf, X, y, scoring="mean_squared_error").mean())


# In[53]:

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

def pred_vs_actual(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    clf.fit(X_train, y_train)
    print clf.best_estimator_
    print "train mse", np.sqrt(mean_squared_error(y_train, clf.predict(X_train)))    
    res = X_test.apply(lambda s: pd.Series({'prediction': clf.predict(s)[0]}) , axis=1)
    res.loc[:, 'actual'] = y_test
    res.loc[:, 'SE'] = res.apply(lambda s: (s['prediction'] - s['actual'])**2 , axis=1)
    print "test mse", np.sqrt(mean_squared_error(res.actual, res.prediction))
    display(res.sort(['SE']).head(3))
    display(res.sort(['SE']).tail(10))
    sns.distplot(res.prediction);
    sns.distplot(res.actual, rug=True);
    return res
    
res = pred_vs_actual(clf, X, y)
print


# In[54]:

from sklearn.externals.six import StringIO
with open("../tree.dot", 'w') as f:
    f = tree.export_graphviz(clf.best_estimator_, feature_names = X.columns, out_file=f)


# In[55]:

from sklearn.ensemble import RandomForestRegressor
clf = grid_search.GridSearchCV(RandomForestRegressor(min_samples_leaf=60, min_samples_split=260, random_state=0), 
                               {'min_samples_leaf': range(60,61,10), 'n_estimators': [1000]})
print np.sqrt(-1.0 * cross_val_score(clf, X, y, scoring="mean_squared_error").mean())


# In[56]:

res = pred_vs_actual(clf, X, y)
print


# In[57]:

forest = clf.best_estimator_
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1][:10]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(10), X.columns[indices], rotation='vertical')
plt.xlim([-1, 10])
plt.show()


# In[ ]:




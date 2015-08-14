
# coding: utf-8

# In[1]:

# Used for selecting the 6 best features per cluster. 
# Assumes data is vectorized + clustered.
# We're using simple f_regression score of each variable vs. the ALSFRS_score, and take the best 6. 


# In[2]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import vectorizer_beta
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model


# In[33]:

proact_data = pd.read_csv('../train_data.csv', sep = '|', index_col=False)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=False)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=False)
print proact_data.shape, slope.shape, clusters.shape
X = pd.merge(clusters, proact_data, on = "SubjectID")
Y = pd.merge(X, slope, on = "SubjectID")

X.head()


# In[28]:

from vectorizer_beta import * 
print func_per_feature


# In[32]:

import vectorizer_beta
selector_per_cluster = {}

for c in clusters['cluster'].unique():
    seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c]
    score_per_feature = {}
    for feature_name, func in func_per_feature.iteritems():
        seg_X_fam = func_per_feature[feature_name](seg_X, feature_name)
        seg_X_fam = seg_X_fam.fillna(seg_X_fam.mean())
        regr = linear_model.LinearRegression()
        regr.fit(seg_X_fam, seg_Y['ALSFRS_slope'])
        score_per_family[family] = np.mean((regr.predict(seg_X_fam) - seg_Y['ALSFRS_slope']) ** 2)
    print c, score_per_family


# In[5]:

def calc(x):
    selector = selector_per_cluster[x['cluster']]
    d = {"feature_ " + str(i): v for i, v in enumerate(selector.transform(x)[0])}
    d['features_list'] = ';'.join(cur_X.columns[selector.get_support()])
    d['cluster'] = int(x['cluster'])
    return pd.Series(d)

for t in ['train', 'test']:
    cur_data = pd.read_csv('../' + t + '_data_vectorized.csv', sep = '|', index_col='SubjectID')
    cur_clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col='SubjectID')
    cur_X = pd.merge(cur_data, cur_clusters, left_index = True, right_index = True)
    res = cur_X.apply(calc, axis = 1)
    res.to_csv('../' + t + '_selected_features.csv',sep='|')
    


# In[ ]:




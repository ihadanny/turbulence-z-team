
# coding: utf-8

# In[7]:

# Used for selecting the 6 best features per cluster. 
# We're using mean squared error of each variable vs. the ALSFRS_score, and take the best 6. 


# In[8]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import vectorizer_beta
from sklearn import linear_model


# In[9]:

proact_data = pd.read_csv('../train_data.csv', sep = '|', index_col=False)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=False)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=False)
X = pd.merge(clusters, proact_data, on = "SubjectID")
Y = pd.merge(clusters, slope, on = "SubjectID")
print Y.shape, X.shape, clusters.shape
X.head()


# In[11]:

from vectorizer_beta import * 
best_features_per_cluster = {}

for c in clusters['cluster'].unique():
    seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c]
    score_per_feature = {}
    for feature_name, func in func_per_feature.iteritems():
        seg_X_fam = func_per_feature[feature_name](seg_X, feature_name)
        seg_Y_fam = pd.merge(seg_Y, seg_X_fam, left_on = 'SubjectID', right_index = True, how='left')
        seg_Y_fam = seg_Y_fam.fillna(seg_Y_fam.mean())
        regr = linear_model.LinearRegression()
        seg_X_fam = seg_Y_fam.drop('ALSFRS_slope', 1)
        regr.fit(seg_X_fam, seg_Y_fam['ALSFRS_slope'])
        score_per_feature[feature_name] = np.mean((regr.predict(seg_X_fam) - seg_Y_fam['ALSFRS_slope']) ** 2)
    print c, score_per_feature
    best_features_per_cluster[c] = sorted(score_per_feature, key=score_per_feature.get)[:6]
best_features_per_cluster


# In[12]:

import pickle 
with open("best_features_per_cluster.pickle", "wb") as output_file:
    pickle.dump(best_features_per_cluster, output_file)


# In[ ]:




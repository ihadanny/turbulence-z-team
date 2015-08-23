
# coding: utf-8

# In[1]:

# Used for selecting the 6 best features per cluster. 
# We're using mean squared error of each variable vs. the ALSFRS_score, and take the best 6. 


# In[2]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model


# In[3]:

#proact_data = pd.read_csv('../train_data.csv', sep = '|', index_col=False)
vectorized_data = pd.read_csv('../train_data_vectorized.csv', sep='|', index_col=0)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=0)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=0)
feature_groups = pickle.load( open('../feature_groups.pickle', 'rb') )

X = clusters.join(vectorized_data)
Y = clusters.join(slope)
X.head()


# In[9]:

#from vectorizer_beta import * 
best_features_per_cluster = {}

for c in clusters['cluster'].unique():
    seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c]
    seg_Y = seg_Y.fillna(seg_Y.mean())
    
    score_per_feature = {}
    
    for feature_group, feature_names in feature_groups.iteritems():
        regr = linear_model.LinearRegression()
        X_feature_fam = seg_X[list(feature_names)]
        regr.fit(X_feature_fam, seg_Y)
        score_per_feature[feature_group] = regr.score(X_feature_fam, seg_Y)
    
    best_features_per_cluster[c] = sorted(sorted(score_per_feature, key=score_per_feature.get)[:6])
    
best_features_per_cluster


# In[10]:

import pickle 
with open("../best_features_per_cluster.pickle", "wb") as output_file:
    pickle.dump(best_features_per_cluster, output_file)


# In[ ]:




# In[ ]:




# In[ ]:




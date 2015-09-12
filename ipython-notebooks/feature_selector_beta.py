
# coding: utf-8

# ## Used for selecting the 6 best features per cluster
# * We're using mean squared error of each variable vs. the ALSFRS_score, and take the best 6. 

# In[69]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from IPython.display import display


# In[17]:

vectorized_data = pd.read_csv('../train_data_vectorized.csv', sep='|', index_col=0)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=0)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=0)
all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )

X = clusters.join(vectorized_data)
Y = clusters.join(slope)
X.head()


# In[18]:

best_features_per_cluster = {}

for c in clusters['cluster'].unique():
    seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c]
    seg_Y = seg_Y.fillna(seg_Y.mean())
    
    score_per_feature = {}
    
    for feature, fm in all_feature_metadata.iteritems():
        regr = linear_model.LinearRegression()
        X_feature_fam = seg_X[list(fm["derived_features"])]
        regr.fit(X_feature_fam, seg_Y)
        score_per_feature[feature] = regr.score(X_feature_fam, seg_Y)
    
    best_features_per_cluster[c] = sorted(sorted(score_per_feature, key=score_per_feature.get)[:6])
    
best_features_per_cluster


# In[19]:

with open("../best_features_per_cluster.pickle", "wb") as output_file:
    pickle.dump(best_features_per_cluster, output_file)


# #Apply the selector 
# leave only the best features per cluster

# In[98]:

for t in ["train", "test"]:
    print t
    df = pd.read_csv('../' + t + '_data.csv', sep = '|', index_col="SubjectID", dtype='unicode')
    print "df", df.shape
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col="SubjectID")
    print "clusters", clusters.shape
    j = df.join(clusters)
    buf, is_first = "", True
    for c, features in best_features_per_cluster.iteritems():
        slice = j[j.cluster == c]
        selected = slice[slice.feature_name.isin(features)]
        print c, slice.shape, selected.shape
        buf += selected.to_csv(sep='|', header = is_first, columns=df.columns)
        is_first = False
    with open('../' + t + '_data_selected.csv','w') as f:
        f.write(buf)


# In[ ]:




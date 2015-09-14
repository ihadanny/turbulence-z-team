
# coding: utf-8

# ## Used for selecting the 6 best features per cluster
# * We're using mean squared error of each variable vs. the ALSFRS_score, and take the best 6. 

# In[1]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from IPython.display import display

from modeling_funcs import *


# In[7]:

vectorized_data = pd.read_csv('../train_data_vectorized.csv', sep='|', index_col=0)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=0)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=0)
all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )

everybody = vectorized_data.join(clusters).join(slope)
Y = everybody[['cluster', 'ALSFRS_slope']]
X = everybody.drop('ALSFRS_slope', 1)


# In[23]:

best_features_per_cluster = get_best_features_per_cluster(X, Y, all_feature_metadata)
best_features_per_cluster


# In[10]:

score_per_feature


# In[11]:

with open("../best_features_per_cluster.pickle", "wb") as output_file:
    pickle.dump(best_features_per_cluster, output_file)


# #Apply the selector 
# leave only the best features per cluster

# In[12]:

for t in ["train", "test"]:
    print t
    df = pd.read_csv('../' + t + '_data.csv', sep = '|', index_col="SubjectID", dtype='unicode')
    print "df", df.shape
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col="SubjectID")
    print "clusters", clusters.shape
    buf = filter_only_selected_features(df, clusters, best_features_per_cluster)
    with open('../' + t + '_data_selected.csv','w') as f:
        f.write(buf)


# In[ ]:





# coding: utf-8

# # Used for predicting ALSFRS_slope
# see https://www.synapse.org/#!Synapse:syn2873386/wiki/ .
# We assumed data is vectorized + clustered + 6 features were selected

# In[1]:

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from vectorizing_funcs import *
from modeling_funcs import *


# ## Revectorize the selected data
# We now reload the metadata and the 6 attributes selected per cluster

# In[2]:

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../train_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../train_data_std.pickle', 'rb') )
best_features_per_cluster = pickle.load( open('../best_features_per_cluster.pickle', 'rb') )


df = pd.read_csv('../train_data_selected.csv', sep='|', index_col=False)
vectorized, _ = vectorize(df, all_feature_metadata)
normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
print normalized.shape
normalized.head()


# In[3]:

slope = pd.read_csv('../train_slope.csv', sep = '|', index_col="SubjectID")
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col="SubjectID")

X = normalized.join(clusters)
Y = slope.join(clusters)

print Y.shape, X.shape, clusters.shape


# ## Train a prediction model per cluster

# In[4]:

model_per_cluster = get_model_per_cluster(X, Y)
    
    


# In[5]:

with open("../model_per_cluster.pickle", "wb") as output_file:
    pickle.dump(model_per_cluster, output_file)


# ## Apply the model on both `train` and `test`

# In[6]:


for t in ['train', 'test']:
    df = pd.read_csv('../' + t + '_data_selected.csv', sep='|', index_col=False)
    vectorized, _ = vectorize(df, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col=0)
    X = normalized.join(clusters)
    pred = X.apply(apply_model, args=[model_per_cluster], axis = 1)
    pred = pred.set_index('SubjectID')
    pred.to_csv('../' + t + '_prediction.csv',sep='|')

pred.head()


# In[ ]:




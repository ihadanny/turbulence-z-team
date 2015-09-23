
# coding: utf-8

# # Used for predicting ALSFRS_slope
# see https://www.synapse.org/#!Synapse:syn2873386/wiki/ .
# We assumed data is vectorized + clustered + 6 features were selected

# In[1]:

from IPython.display import display

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from vectorizing_funcs import *
from modeling_funcs import *


# ## Revectorize the selected data
# We now reload the metadata and the 6 attributes selected per cluster

# In[12]:

# load all metadata
all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../all_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../all_data_std.pickle', 'rb') )
best_features_per_cluster = pickle.load( open('../best_features_per_cluster.pickle', 'rb') )

# reload the data imputed only for the 6 selected attributes per cluster
df = pd.read_csv('../all_data_selected.csv', sep='|', index_col=False, dtype="unicode")
vectorized, _ = vectorize(df, all_feature_metadata)
normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
print normalized.shape
normalized.head()


# In[13]:

d = normalized.describe().T
d[d['std'] > 0.0].sort("std", ascending=False)


# In[14]:

slope = pd.read_csv('../all_slope.csv', sep = '|', index_col="SubjectID")
clusters = pd.read_csv('../all_forest_clusters.csv', sep = '|', index_col="SubjectID")

clusters.index = clusters.index.astype(str)
slope.index = slope.index.astype(str)
normalized.index = normalized.index.astype(str)

X = normalized.join(clusters)
Y = slope.join(clusters)

print Y.shape, X.shape, clusters.shape
print clusters.groupby('cluster').size()
display(Y.head(3))


# ## Train a prediction model per cluster

# In[17]:


model_per_cluster = get_model_per_cluster(X, Y)
    


# In[18]:

with open("../model_per_cluster.pickle", "wb") as output_file:
    pickle.dump(model_per_cluster, output_file)


# ## Apply the model on both `train` and `test`

# In[20]:


for t in ['all', 'test']:
    print t
    df = pd.read_csv('../' + t + '_data_selected.csv', sep='|', index_col=False, dtype="unicode")
    vectorized, _ = vectorize(df, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    clusters = pd.read_csv('../' + t + '_forest_clusters.csv', sep = '|', index_col=0)

    clusters.index = clusters.index.astype(str)
    normalized.index = normalized.index.astype(str)

    X = normalized.join(clusters)
    print X.groupby('cluster').size()
    pred = X.apply(apply_model, args=[model_per_cluster], axis = 1)
    pred.to_csv('../' + t + '_prediction.csv',sep='|')

pred.head()


# In[ ]:




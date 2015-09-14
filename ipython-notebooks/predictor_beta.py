
# coding: utf-8

# # Used for predicting ALSFRS_slope
# see https://www.synapse.org/#!Synapse:syn2873386/wiki/ .
# We assumed data is vectorized + clustered + 6 features were selected

# In[11]:

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

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../all_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../all_data_std.pickle', 'rb') )
best_features_per_cluster = pickle.load( open('../best_features_per_cluster.pickle', 'rb') )


df = pd.read_csv('../all_data_selected.csv', sep='|', index_col=False, dtype="unicode")
vectorized, _ = vectorize(df, all_feature_metadata)
normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
print normalized.shape
normalized.head()


# In[13]:

normalized.describe().T.sort("std", ascending=False)


# In[14]:

slope = pd.read_csv('../all_slope.csv', sep = '|', index_col="SubjectID")
clusters = pd.read_csv('../all_kmeans_clusters.csv', sep = '|', index_col="SubjectID")

clusters.index = clusters.index.astype(str)
slope.index = slope.index.astype(str)
normalized.index = normalized.index.astype(str)

X = normalized.join(clusters)
Y = slope.join(clusters)

print Y.shape, X.shape, clusters.shape
display(Y.head(3))


# ## Train a prediction model per cluster

# In[17]:

from sklearn import linear_model
import numpy as np

def get_model_per_cluster(X, Y):
    model_per_cluster = {}
    for c in X.cluster.unique():    
        X_cluster = X[X.cluster==c]
        Y_cluster = Y[Y.cluster == c].ALSFRS_slope
        regr = linear_model.LinearRegression()
        regr.fit(X_cluster, Y_cluster)

        print 'cluster: %d size: %s' % (c, Y_cluster.shape)
        print "\t RMS error (0 is perfect): %.2f" % np.sqrt(np.mean(
            (regr.predict(X_cluster) - Y_cluster) ** 2))
        print('\t explained variance score (1 is perfect): %.2f' % regr.score(X_cluster, Y_cluster))
        print "3 sample predictions: ", regr.predict(X_cluster)[:3]
        model_per_cluster[c] = {"train_data_means": X_cluster.mean(), "model" : regr}
    return model_per_cluster

model_per_cluster = get_model_per_cluster(X, Y)
    


# In[18]:

with open("../model_per_cluster.pickle", "wb") as output_file:
    pickle.dump(model_per_cluster, output_file)


# ## Apply the model on both `train` and `test`

# In[22]:


for t in ['all', 'test']:
    print t
    df = pd.read_csv('../' + t + '_data_selected.csv', sep='|', index_col=False, dtype="unicode")
    vectorized, _ = vectorize(df, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col=0)

    clusters.index = clusters.index.astype(str)
    normalized.index = normalized.index.astype(str)

    X = normalized.join(clusters)
    pred = X.apply(apply_model, args=[model_per_cluster], axis = 1)
    pred.to_csv('../' + t + '_prediction.csv',sep='|')

pred.head()


# In[ ]:




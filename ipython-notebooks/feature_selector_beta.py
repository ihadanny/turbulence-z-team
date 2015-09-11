
# coding: utf-8

# ## Used for selecting the 6 best features per cluster
# * We're using mean squared error of each variable vs. the ALSFRS_score, and take the best 6. 

# In[16]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from vectorizing_funcs import *


# In[17]:

#proact_data = pd.read_csv('../train_data.csv', sep = '|', index_col=False)
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


# ## Run selector.sh
# As specified in the challenge: The output_file_path must have the following format:
# * First line: the cluster identifier for that patient
# * Following lines: the selected features selected for that specific single patient, using the same format as the input data. A maximum of 6 features are allowed.

# In[42]:

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../train_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../train_data_std.pickle', 'rb') )
clustering_model = pickle.load( open('../clustering_model.pickle', 'rb') )


t = "test"
df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
for subj in df.SubjectID.unique()[:5]:
    df_subj = df[df.SubjectID == subj]
    vectorized, _ = vectorize(df_subj, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    cluster_data = normalized[clustering_model["columns"]]
    c = clustering_model["model"].predict(cluster_data)[0]
    print "cluster: ", c
    selected = df_subj[df_subj.feature_name.isin(best_features_per_cluster[c])]
    print selected.to_csv(sep='|', index = False)



# In[ ]:




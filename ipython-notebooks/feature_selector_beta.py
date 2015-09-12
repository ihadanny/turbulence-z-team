
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


# ## Run selector.sh
# As specified in the challenge - we must run our selector logic subject by subject.
# 
# The output_file_path must have the following format:
# * First line: the cluster identifier for that patient
# * Following lines: the selected features selected for that specific single patient, using the same format as the input data. A maximum of 6 features are allowed.

# In[10]:

import pickle
import pandas as pd
from vectorizing_funcs import *

all_feature_metadata = pickle.load( open('../all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open('../train_data_means.pickle', 'rb') )
train_data_std = pickle.load( open('../train_data_std.pickle', 'rb') )
clustering_model = pickle.load( open('../clustering_model.pickle', 'rb') )
best_features_per_cluster = pickle.load( open('../best_features_per_cluster.pickle', 'rb') )


t = "test"
df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
for subj in df.SubjectID.unique()[:3]:
    df_subj = df[df.SubjectID == subj]
    vectorized, _ = vectorize(df_subj, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    cluster_data = normalized[clustering_model["columns"]]
    c = clustering_model["model"].predict(cluster_data)[0]
    buf = "cluster: %d\n" % c
    selected = df_subj[df_subj.feature_name.isin(best_features_per_cluster[c])]
    buf += selected.to_csv(sep='|', index = False, header = False)
    print buf
    with open('../selected_' + subj + ".txt", "wb") as f:
        f.write(buf)


# In[ ]:




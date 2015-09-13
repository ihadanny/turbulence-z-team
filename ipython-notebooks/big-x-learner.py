
# coding: utf-8

# ## Builds all our models x-validated
# 

# In[7]:

from IPython.display import display

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from StringIO import StringIO

from vectorizing_funcs import *
from modeling_funcs import *


# In[8]:

df = pd.read_csv('../all_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
slope = pd.read_csv('../all_slope.csv', sep = '|', index_col="SubjectID")
slope.index = slope.index.astype(str)

print "df: ", df.shape, df.SubjectID.unique().size
print "slope: ", slope.shape, slope.index.unique().size
display(df.head(2))
display(slope.head(2))


# In[9]:

all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))


# In[10]:

def apply_on_test(test_data, all_feature_metadata, train_data_means, train_data_std, 
                 clustering_columns, kmeans, best_features_per_cluster, model_per_cluster):
    
    # Vectorizing
    vectorized, _ = vectorize(test_data, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    
    # Clustering
    for_clustering = normalized[clustering_columns]
    clusters = pd.DataFrame(index = for_clustering.index.astype(str))
    clusters['cluster'] = kmeans.predict(for_clustering)
    X = normalized.join(clusters)
    
    buf = filter_only_selected_features(test_data.set_index("SubjectID"), clusters,                                         best_features_per_cluster)    
    s_df = pd.read_csv(StringIO(buf), sep='|', index_col=False, dtype='unicode')
    s_vectorized, _ = vectorize(s_df, all_feature_metadata)
    s_normalized, _ = normalize(s_vectorized, all_feature_metadata, train_data_means, train_data_std)    
    s_X = s_normalized.join(clusters)    

    pred = s_X.apply(apply_model, args=[model_per_cluster], axis = 1)
    return pred
    


# In[11]:

from sklearn.cross_validation import KFold
kf = KFold(df.SubjectID.unique().size, n_folds=2)
fold = 0
for train, test in kf:
    train_data = df[df.SubjectID.isin(df.SubjectID.unique()[train])]
    test_data = df[df.SubjectID.isin(df.SubjectID.unique()[test])]
    print "fold: %d" % fold
    print "train_data: ", train_data.shape, train_data.SubjectID.unique().size,             train_data.SubjectID.min(), train_data.SubjectID.max()
    
    # Vectorizing
    all_feature_metadata = learn_to_dummies_model(train_data, all_feature_metadata)
    vectorized, all_feature_metadata = vectorize(train_data, all_feature_metadata)
    train_data_means = vectorized.mean()
    train_data_std = vectorized.std()            
    normalized, all_feature_metadata = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    
    # Clustering
    for_clustering = normalized[clustering_columns]
    kmeans = KMeans(init='k-means++', n_clusters=3)
    #Note we must convert to str to join with slope later
    clusters = pd.DataFrame(index = for_clustering.index.astype(str))
    clusters['cluster'] = kmeans.fit_predict(for_clustering)
    X = normalized.join(clusters)
    Y = slope.join(clusters)

    best_features_per_cluster = get_best_features_per_cluster(X, Y, all_feature_metadata)
    print "best_features_per_cluster: ", best_features_per_cluster 
    buf = filter_only_selected_features(train_data.set_index("SubjectID"), clusters,                                         best_features_per_cluster)
    
    s_df = pd.read_csv(StringIO(buf), sep='|', index_col=False, dtype='unicode')
    s_vectorized, _ = vectorize(s_df, all_feature_metadata)
    s_normalized, _ = normalize(s_vectorized, all_feature_metadata, train_data_means, train_data_std)    
    s_X = s_normalized.join(clusters)
    
    model_per_cluster = get_model_per_cluster(s_X, Y)

    pred = apply_on_test(train_data, all_feature_metadata, train_data_means, train_data_std, 
                 clustering_columns, kmeans, best_features_per_cluster, model_per_cluster)
    res = pred.join(slope)
    print "Train root mean square error (0 is perfect): %.2f" % np.sqrt(np.mean(
        (res.prediction - res.ALSFRS_slope) ** 2))


    fold += 1



# In[ ]:





# coding: utf-8

# ## Builds all our models x-validated
# 

# In[1]:

from IPython.display import display

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from StringIO import StringIO
from sklearn import metrics
from sklearn.cross_validation import KFold

from vectorizing_funcs import *
from modeling_funcs import *


# In[2]:

df = pd.read_csv('../all_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
slope = pd.read_csv('../all_slope.csv', sep = '|', index_col="SubjectID")
slope.index = slope.index.astype(str)

print "df: ", df.shape, df.SubjectID.unique().size
print "slope: ", slope.shape, slope.index.unique().size
display(df.head(2))
display(slope.head(2))


# In[3]:

clustering_columns = [u'Asian',
       u'mouth_last', u'mouth_mean_slope',u'hands_last',
       u'hands_mean_slope',u'onset_delta_last', u'ALSFRS_Total_last',
       u'ALSFRS_Total_mean_slope', u'fvc_percent_mean_slope', 
                     u'respiratory_last', u'respiratory_mean_slope']


# In[4]:

def apply_on_test(test_data, all_feature_metadata, train_data_means, train_data_std, 
                 clustering_columns, kmeans, best_features_per_cluster, model_per_cluster):
    
    # Vectorizing
    vectorized, _ = vectorize(test_data, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    
    print "applying on: ", normalized.shape
    
    # Clustering
    
    for_clustering = normalized[clustering_columns]
    clusters = pd.DataFrame(index = for_clustering.index.astype(str))
    clusters['cluster'] = kmeans.predict(for_clustering)
    print "applied cluster cnt: ", np.bincount(clusters.cluster)

    X = normalized.join(clusters)
    
    buf = filter_only_selected_features(test_data.set_index("SubjectID"), clusters,                                         best_features_per_cluster)    
    s_df = pd.read_csv(StringIO(buf), sep='|', index_col=False, dtype='unicode')
    s_vectorized, _ = vectorize(s_df, all_feature_metadata)
    s_normalized, _ = normalize(s_vectorized, all_feature_metadata, train_data_means, train_data_std)    
    input_for_model = s_normalized.join(clusters)    
    
    pred = input_for_model.apply(apply_model, args=[model_per_cluster], axis = 1)
    return input_for_model, pred
    


# In[5]:

def train_it(train_data, my_n_clusters):
        global ts_funcs_to_features
        # Prepare metadata
        ts_funcs_to_features = add_frequent_lab_tests_to_ts_features(train_data, ts_funcs_to_features)
        all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
        all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))
        all_feature_metadata = learn_to_dummies_model(train_data, all_feature_metadata)
        
        # Vectorizing
        vectorized, all_feature_metadata = vectorize(train_data, all_feature_metadata)
        train_data_means = vectorized.mean()
        train_data_std = vectorized.std()            
        normalized, all_feature_metadata = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)

        print "train_data: ", normalized.shape
        
        # Clustering
        for_clustering = normalized[clustering_columns]
        kmeans = KMeans(init='k-means++', n_clusters=my_n_clusters)
        # Note we must convert to str to join with slope later
        clusters = pd.DataFrame(index = for_clustering.index.astype(str))
        clusters['cluster'] = kmeans.fit_predict(for_clustering)
        print "train cluster cnt: ", np.bincount(clusters.cluster)

        X = normalized.join(clusters)
        Y = slope.join(clusters)

        best_features_per_cluster = stepwise_best_features_per_cluster(X, Y, all_feature_metadata)
        print "best_features_per_cluster: ", best_features_per_cluster 
        buf = filter_only_selected_features(train_data.set_index("SubjectID"), clusters,                                             best_features_per_cluster)

        s_df = pd.read_csv(StringIO(buf), sep='|', index_col=False, dtype='unicode')
        s_vectorized, _ = vectorize(s_df, all_feature_metadata)
        s_normalized, _ = normalize(s_vectorized, all_feature_metadata, train_data_means, train_data_std)    
        s_X = s_normalized.join(clusters)
        
        model_per_cluster = get_model_per_cluster(s_X, Y)
        
        return all_feature_metadata, train_data_means, train_data_std,                      kmeans, best_features_per_cluster, model_per_cluster


# In[6]:

from datetime import datetime

def train_and_test(df, slope, my_n_clusters=3):
    kf = KFold(df.SubjectID.unique().size, n_folds=3)
    fold, test_rmse, train_rmse = 0, 0.0, 0.0

    for train, test in kf:
        train_data = df[df.SubjectID.isin(df.SubjectID.unique()[train])]
        test_data = df[df.SubjectID.isin(df.SubjectID.unique()[test])]
        print
        print "*"*30
        print "fold: %d" % fold
        tick = datetime.now()
        
        all_feature_metadata, train_data_means, train_data_std,                      kmeans, best_features_per_cluster, model_per_cluster = train_it(train_data, my_n_clusters)

        input_for_model, pred = apply_on_test(train_data, all_feature_metadata, train_data_means, train_data_std, 
                     clustering_columns, kmeans, best_features_per_cluster, model_per_cluster)
        res = pred.join(slope)
        train_rmse += np.sqrt(np.mean((res.prediction - res.ALSFRS_slope) ** 2))

        input_for_model, pred = apply_on_test(test_data, all_feature_metadata, train_data_means, train_data_std, 
                     clustering_columns, kmeans, best_features_per_cluster, model_per_cluster)
        res = pred.join(slope)
        test_rmse += np.sqrt(np.mean((res.prediction - res.ALSFRS_slope) ** 2))
        
        input_for_model.to_csv('../x_results/test_%d_input_for_model.csv' % fold,sep='|')
        res.to_csv('../x_results/test_%d_prediction.csv' % fold,sep='|')

        fold += 1
        print "fold RMS Error train, test: ", train_rmse / fold, test_rmse / fold

        tock = datetime.now()   
        diff = tock - tick 
        print "minutes for fold: ", diff.seconds / 60

            
    print "X-validated RMS Error train, test: ", train_rmse / kf.n_folds, test_rmse / kf.n_folds



# In[7]:

for n_clusters in range(2, 4):
    print "*"*60
    print "*"*60
    train_and_test(df, slope, n_clusters)


# In[ ]:





# coding: utf-8

# In[17]:

# Used for predicting ALSFRS_slope (see https://www.synapse.org/#!Synapse:syn2873386/wiki/)
# Assumed data is vectorized + clustered + 6 features were selected


# In[18]:

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model


# In[19]:

proact_data = pd.read_csv('../train_data.csv', sep = '|', index_col=False)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=False)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=False)
with open("best_features_per_cluster.pickle", "rb") as input_file:
    best_features_per_cluster = pickle.load(input_file)

X = pd.merge(clusters, proact_data, on = "SubjectID")
Y = pd.merge(clusters, slope, on = "SubjectID")

print Y.shape, X.shape, clusters.shape


# In[32]:

from vectorizer_beta import * 
model_per_cluster = {}

def vectorize(clusters, seg_X, c):
    seg_vectorized_X = clusters[clusters['cluster'] == c]
    for feature_name in best_features_per_cluster[c]:
        seg_X_feature = func_per_feature[feature_name](seg_X, feature_name)
        seg_vectorized_X = pd.merge(seg_vectorized_X, seg_X_feature, left_on = 'SubjectID', right_index = True, how='left')
    return seg_vectorized_X

for c in clusters['cluster'].unique():
    seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c]
    seg_vectorized_X = vectorize(clusters, seg_X, c)
    train_data_means = seg_vectorized_X.mean()
    seg_vectorized_X = seg_vectorized_X.fillna(train_data_means) 
    seg_Y = pd.merge(seg_Y, seg_vectorized_X, on = ['SubjectID', 'cluster'])
    regr = linear_model.LinearRegression()
    regr.fit(seg_Y.drop('ALSFRS_slope', 1), seg_Y['ALSFRS_slope'])
    print 'cluster: %d size: %s' % (c, seg_Y.shape)
    print "Mean square error (0 is perfect): %.2f" % np.mean(
        (regr.predict(seg_Y.drop('ALSFRS_slope', 1)) - seg_Y['ALSFRS_slope']) ** 2)
    print('Explained variance score (1 is perfect): %.2f' % regr.score(seg_Y.drop('ALSFRS_slope', 1), seg_Y['ALSFRS_slope']))
    
    model_per_cluster[c] = {"train_data_means": train_data_means, "model" : regr}
    
    


# In[52]:

def calc(x):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    pred = float(model.predict(x))
    return pd.Series({'SubjectID': int(x['SubjectID']), 'prediction':pred, 'cluster': int(c), 'features_list': ";".join(best_features_per_cluster[c])})

for t in ['train', 'test']:
    proact_data = pd.read_csv('../' + t + '_data.csv', sep = '|', index_col=False)
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col=False)
    X = pd.merge(clusters, proact_data, on = "SubjectID")
    final = None
    for c in clusters['cluster'].unique():
        seg_X = X[X['cluster'] == c]
        seg_vectorized_X = vectorize(clusters, seg_X, c)
        train_data_means = model_per_cluster[c]['train_data_means']
        seg_vectorized_X = seg_vectorized_X.fillna(train_data_means) 
        pred_c = seg_vectorized_X.apply(calc, axis = 1)
        pred_c = pred_c.set_index('SubjectID')
        if final is not None:
            final = final.append(pred_c)
        else:
            final = pred_c
    final.to_csv('../' + t + '_prediction.csv',sep='|', columns=['prediction', 'cluster', 'features_list'])
final.head()


# In[ ]:




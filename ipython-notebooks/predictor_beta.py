
# coding: utf-8

# In[1]:

# Used for predicting ALSFRS_slope (see https://www.synapse.org/#!Synapse:syn2873386/wiki/)
# Assumed data is vectorized + clustered + 6 features were selected


# In[2]:

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model


# In[60]:

# proact_data = pd.read_csv('../train_data.csv', sep = '|', index_col=0)
slope = pd.read_csv('../train_slope.csv', sep = '|', index_col=0)
clusters = pd.read_csv('../train_kmeans_clusters.csv', sep = '|', index_col=0)
vectorized = pd.read_csv('../train_data_vectorized.csv', sep='|', index_col=0)
with open("../feature_groups.pickle", "rb") as input_file:
    feature_groups = pickle.load(input_file)
with open("../best_features_per_cluster.pickle", "rb") as input_file:
    best_features_per_cluster = pickle.load(input_file)

X = clusters.join(vectorized)
Y = clusters.join(slope)

print Y.shape, X.shape, clusters.shape


# In[61]:

#from vectorizer_beta import * 
model_per_cluster = {}

# def vectorize(clusters, seg_X, c):
#     seg_vectorized_X = clusters[clusters['cluster'] == c]
#     for feature_name in best_features_per_cluster[c]:
#         seg_X_feature = func_per_feature[feature_name](seg_X, feature_name)
#         seg_vectorized_X = pd.merge(seg_vectorized_X, seg_X_feature, left_on = 'SubjectID', right_index = True, how='left')
#     return seg_vectorized_X

clusters_vectorized = clusters.join(vectorized)
for c in clusters.cluster.unique():
    best_feature_groups = best_features_per_cluster[c]
    best_features = list(reduce( lambda x,y: x|y, [ feature_groups[g] for g in best_feature_groups ]))
    
    X = clusters_vectorized[clusters_vectorized.cluster==c][best_features]
    Y_data = Y[Y['cluster'] == c].ALSFRS_slope
    regr = linear_model.LinearRegression()
    regr.fit(X, Y_data)

    print 'cluster: %d size: %s' % (c, Y.shape)
    print 'Best feature groups: ', best_feature_groups
    print "Mean square error (0 is perfect): %.2f" % np.mean(
        (regr.predict(X) - Y_data) ** 2)
    print('Explained variance score (1 is perfect): %.2f' % regr.score(X, Y_data))
    print ""
    model_per_cluster[c] = {"train_data_means": X.mean(), "model" : regr}
    
    


# In[73]:

def calc(x):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    best_feature_groups = best_features_per_cluster[c]
    best_features = list(reduce( lambda x,y: x|y, [ feature_groups[g] for g in best_feature_groups ]))
    x = x[best_features]
    pred = float(model.predict(x))
    return pd.Series({'SubjectID': int(x.name), 'prediction':pred, 'cluster': int(c), 'features_list': ";".join(best_features_per_cluster[c])})

for t in ['train', 'test']:
    vectorized_data = pd.read_csv('../' + t + '_data_vectorized.csv', sep = '|', index_col=0)
    clusters = pd.read_csv('../' + t + '_kmeans_clusters.csv', sep = '|', index_col=0)
    X = clusters.join(vectorized_data)
    final = None
    for c in clusters['cluster'].unique():
        seg_X = X[X['cluster'] == c]
        train_data_means = model_per_cluster[c]['train_data_means']
        seg_vectorized_X = seg_X.fillna(train_data_means) 
        pred_c = seg_X.apply(calc, axis = 1)
        pred_c = pred_c.set_index('SubjectID')
        if final is not None:
            final = final.append(pred_c)
        else:
            final = pred_c
    final.to_csv('../' + t + '_prediction.csv',sep='|', columns=['prediction', 'cluster', 'features_list'])
    
    print t, ' mean squared errors - ', np.mean((final['prediction'] - Y['ALSFRS_slope']) ** 2)    

final.head()


# In[ ]:




# In[ ]:




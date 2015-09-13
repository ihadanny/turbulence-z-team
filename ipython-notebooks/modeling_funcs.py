
# coding: utf-8

# ## Clustering hard-coded columns

# In[1]:

clustering_columns = [
    'ALSFRS_Total_last',
    'ALSFRS_Total_mean_slope',
    'weight_mean', 
    'weight_pct_diff',
    'Age_last',
    
    'onset_delta_last',
    'Albumin_last',
    'Creatinine_last',
    'fvc_percent_pct_diff',
    'bp_systolic_mean',
        
]


# ## Feature selection
# We currently rank each feature family by regressing with it alone and comparing the regression score

# In[2]:

from sklearn import linear_model

def get_best_features_per_cluster(X, Y, all_feature_metadata):
    best_features_per_cluster = {}
    for c in X['cluster'].unique():
        seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c]
        seg_Y = seg_Y.fillna(seg_Y.mean())

        score_per_feature = {}

        for feature, fm in all_feature_metadata.iteritems():
            regr = linear_model.LinearRegression()
            X_feature_fam = seg_X[list(fm["derived_features"])]
            regr.fit(X_feature_fam, seg_Y)
            score_per_feature[feature] = regr.score(X_feature_fam, seg_Y)

        best_features_per_cluster[c] = sorted(sorted(score_per_feature, key=score_per_feature.get)[:6])
    return best_features_per_cluster


# In[3]:

def filter_only_selected_features(df, clusters, best_features_per_cluster, debug=False): 
    j = df.join(clusters)
    buf, is_first = "", True
    for c, features in best_features_per_cluster.iteritems():
        slice = j[j.cluster == c]
        selected = slice[slice.feature_name.isin(features)]
        if debug:
            print c, slice.shape, " --> ", selected.shape
        buf += selected.to_csv(sep='|', header = is_first, columns=df.columns)
        is_first = False
    return buf


# ## Prediction
# We use simple linear regression

# In[4]:

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
        print "root mean square error (0 is perfect): %.2f" % np.sqrt(np.mean(
            (regr.predict(X_cluster) - Y_cluster) ** 2))
        print('Explained variance score (1 is perfect): %.2f' % regr.score(X_cluster, Y_cluster))
        print ""
        model_per_cluster[c] = {"train_data_means": X_cluster.mean(), "model" : regr}
    return model_per_cluster


# In[5]:

import pandas as pd

def apply_model(x, model_per_cluster):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    pred = float(model.predict(x))
    return pd.Series({'prediction':pred, 'cluster': int(c)})


# In[ ]:




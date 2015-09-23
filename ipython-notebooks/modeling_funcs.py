
# coding: utf-8

# ## Clustering hard-coded columns

# In[1]:

clustering_columns = [u'Asian', u'Black', u'Hispanic', u'Other', u'Unknown', u'White',
       u'mouth_last', u'mouth_mean_slope',u'hands_last',
       u'hands_mean_slope',u'onset_delta_last', u'ALSFRS_Total_last',
       u'ALSFRS_Total_mean_slope',u'BMI_last', u'fvc_percent_mean_slope', 
                     u'respiratory_last', u'respiratory_mean_slope']


# ## Feature selection
# We currently rank each feature family by regressing with it alone and comparing the regression score

# In[1]:

from sklearn import linear_model
import operator
import time
from sklearn.linear_model import LassoCV, LassoLarsCV

def get_best_features_per_cluster(X, Y, all_feature_metadata):
    best_features_per_cluster = {}
    for c in X['cluster'].unique():
        seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c].ALSFRS_slope
        seg_Y = seg_Y.fillna(seg_Y.mean())

        score_per_feature = {}

        for feature, fm in all_feature_metadata.iteritems():
            regr = linear_model.LinearRegression()
            X_feature_fam = seg_X[list(fm["derived_features"])]
            regr.fit(X_feature_fam, seg_Y)
            score_per_feature[feature] = np.sqrt(np.mean((regr.predict(X_feature_fam) - seg_Y) ** 2))
            regr.score(X_feature_fam, seg_Y)
        best_features_per_cluster[c] = [k for k,v in sorted(score_per_feature.items(), key=operator.itemgetter(1))[:6]]
    return best_features_per_cluster


# In[2]:

def stepwise_best_features_per_cluster(X, Y, all_feature_metadata):
    best_features_per_cluster = {}
    for c in sorted(X['cluster'].unique()):
        seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c].ALSFRS_slope
        print "cluster:", c, "with size:", seg_X.shape, "with mean target:", seg_Y.mean(), "std:", seg_Y.std()
        seg_Y = seg_Y.fillna(seg_Y.mean())
        
        model = LassoCV(cv=5).fit(seg_X, seg_Y)
        print "best we can do with all features:", np.sqrt(np.mean((model.predict(seg_X) - seg_Y) ** 2))

        selected_fams = set()
        selected_derived = set()
        for i in range(6):
            score_per_family = {}
            t1 = time.time()
            for family, fm in all_feature_metadata.iteritems():
                if family not in selected_fams:                    
                    X_feature_fam = seg_X[list(selected_derived) + list(fm["derived_features"])]
                    model = LassoCV(cv=5).fit(X_feature_fam, seg_Y)
                    score_per_family[family] = np.sqrt(np.mean((model.predict(X_feature_fam) - seg_Y) ** 2))
            t_lasso_cv = time.time() - t1
            best_fam = sorted(score_per_family.items(), key=operator.itemgetter(1))[0]
            print "adding best family:", best_fam, "time:", t_lasso_cv
            selected_fams.add(best_fam[0])
            selected_derived.update(all_feature_metadata[best_fam[0]]["derived_features"])
        best_features_per_cluster[c] = list(selected_fams)                          
    return best_features_per_cluster


# In[ ]:

from sklearn.ensemble import RandomForestRegressor
def backward_best_features_per_cluster(X, Y, all_feature_metadata):
    best_features_per_cluster = {}
    for c in sorted(X['cluster'].unique()):
        seg_X, seg_Y = X[X['cluster'] == c], Y[Y['cluster'] == c].ALSFRS_slope
        print "cluster:", c, "with size:", seg_X.shape, "with mean target:", seg_Y.mean(), "std:", seg_Y.std()
        seg_Y = seg_Y.fillna(seg_Y.mean())
        
        model = RandomForestRegressor(min_samples_leaf=60, random_state=0, n_estimators=1000).fit(seg_X, seg_Y)
        print "best we can do with all features:", np.sqrt(np.mean((model.predict(seg_X) - seg_Y) ** 2))

        selected_fams = set(all_feature_metadata.keys())
        selected_derived = set([])
        for fam in selected_fams:
            selected_derived.update([der for der in all_feature_metadata[fam]['derived_features']])
        while len(selected_fams) > 6:
            score_per_family = {}
            t1 = time.time()
            for family, fm in all_feature_metadata.iteritems():
                if family in selected_fams:
                    X_feature_fam = seg_X[list(selected_derived - set(fm["derived_features"]))]
                    model = RandomForestRegressor(min_samples_leaf=60, random_state=0, n_estimators=1000).fit(
                        X_feature_fam, seg_Y)
                    score_per_family[family] = np.sqrt(np.mean((model.predict(X_feature_fam) - seg_Y) ** 2))
            t_lasso_cv = time.time() - t1
            worst_fam = sorted(score_per_family.items(), key=operator.itemgetter(1), reverse=True)[0]
            print "removing worst family:", worst_fam, "time:", t_lasso_cv
            selected_fams.remove(worst_fam[0])
            selected_derived = set([])
            for fam in selected_fams:
                selected_derived.update([der for der in all_feature_metadata[fam]['derived_features']])
        best_features_per_cluster[c] = list(selected_fams)                          
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
from sklearn.linear_model import LassoCV, LassoLarsCV
import numpy as np

def get_model_per_cluster(X, Y):
    model_per_cluster = {}
    for c in X.cluster.unique():    
        X_cluster = X[X.cluster==c]
        Y_cluster = Y[Y.cluster == c].ALSFRS_slope
        
        regr = LassoCV(cv=5)
        regr.fit(X_cluster, Y_cluster)

        print 'cluster: %d size: %s' % (c, Y_cluster.shape)
        print "\t RMS error (0 is perfect): %.2f" % np.sqrt(np.mean(
            (regr.predict(X_cluster) - Y_cluster) ** 2))
        print('\t explained variance score (1 is perfect): %.2f' % regr.score(X_cluster, Y_cluster))
        print "3 sample predictions: ", regr.predict(X_cluster)[:3]
        model_per_cluster[c] = {"cluster_train_data_means": X_cluster.mean(), "model" : regr}
    return model_per_cluster


# In[5]:

import pandas as pd

def apply_model(x, model_per_cluster):
    c = x['cluster']
    model = model_per_cluster[c]['model']
    pred = float(model.predict(x))
    return pd.Series({'prediction':pred, 'cluster': int(c)})


# In[ ]:




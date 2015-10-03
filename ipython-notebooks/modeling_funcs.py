
# coding: utf-8

# In[ ]:

from vectorizing_funcs import *


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
        
        model = RandomForestRegressor(min_samples_leaf=60, random_state=0, n_estimators=1000)
        #model = LassoCV(cv=5)
        model = model.fit(seg_X, seg_Y)
        
        print "best we can do with all features:", np.sqrt(np.mean((model.predict(seg_X) - seg_Y) ** 2))
        print "using model:", model

        selected_fams = set()
        selected_derived = set()
        for i in range(6):
            score_per_family = {}
            t1 = time.time()
            for family, fm in all_feature_metadata.iteritems():
                if family not in selected_fams:                    
                    X_feature_fam = seg_X[list(selected_derived) + list(fm["derived_features"])]
                    model = RandomForestRegressor(min_samples_leaf=60, random_state=0, n_estimators=1000)
                    #model = LassoCV(cv=5)
                    model = model.fit(X_feature_fam, seg_Y)
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
import scipy

def get_model_per_cluster(X, Y):
    model_per_cluster = {}
    for c in X.cluster.unique():    
        X_cluster = X[X.cluster==c]
        Y_true = Y[Y.cluster == c].ALSFRS_slope
        
        regr = LassoCV(cv=5)
        regr.fit(X_cluster, Y_true)

        print 'cluster: %d size: %s' % (c, Y_true.shape)
        Y_predict = regr.predict(X_cluster)
        print "\t RMS error (0 is perfect): %.2f" % np.sqrt(np.mean(
            (Y_predict - Y_true) ** 2))
        regression_SS = ((Y_predict - Y_true) ** 2).sum()
        residual_SS =((Y_true - Y_true.mean()) ** 2).sum()
        print '\t coefficient of determination R^2 = %.2f ' % (1.0 - regression_SS/residual_SS) # regr.score(X_cluster, Y_true)
        cov = sum((Y_predict - Y_predict.mean())*(Y_true - Y_true.mean()))
        Y_predict_std = np.sqrt(sum((Y_predict - Y_predict.mean())**2))
        Y_true_std = np.sqrt(sum((Y_true - Y_true.mean())**2))
        print '\t pearson correlation r = %.2f ' % (cov/(Y_predict_std*Y_true_std)) # scipy.stats.pearsonr(Y_predict, Y_true)[0]
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


# In[1]:

from sklearn import cross_validation, grid_search
from sklearn.ensemble import RandomForestRegressor
from StringIO import StringIO

def train_it(train_data, slope, my_n_clusters):
        global ts_funcs_to_features
        # Prepare metadata
        ts_funcs_to_features = add_frequent_lab_tests_to_ts_features(train_data, ts_funcs_to_features)
        all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
        all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))
        all_feature_metadata = learn_to_dummies_model(train_data, all_feature_metadata)
        
        # Vectorizing
        vectorized, all_feature_metadata = vectorize(train_data, all_feature_metadata)
        train_data_medians = vectorized.median()
        train_data_mads = (vectorized - train_data_medians).abs().median()
        train_data_std = vectorized.std()
        cleaned = clean_outliers(vectorized, all_feature_metadata, train_data_medians, train_data_mads, train_data_std)
        train_data_means = cleaned.mean()
        train_data_std = cleaned.std()            
        normalized, all_feature_metadata = normalize(cleaned, all_feature_metadata, train_data_means, train_data_std)

        everybody = normalized.join(slope)
        everybody = everybody[~np.isnan(everybody.ALSFRS_slope)]

        X = everybody.drop(['ALSFRS_slope'], 1)
        Y = everybody[['ALSFRS_slope']]
        print "train_data: ", X.shape, Y.shape
        
        everybody.to_csv('../x_results/input_for_forest_selector.csv', sep='|')

        # Clustering
        #for_clustering = normalized[clustering_columns]
        #kmeans = KMeans(init='k-means++', n_clusters=my_n_clusters)
        #clusters['cluster'] = kmeans.fit_predict(for_clustering)

        forest = RandomForestRegressor(min_samples_leaf=60, min_samples_split=260, random_state=0, 
                               n_estimators=1000)
        forest.fit(X, Y.ALSFRS_slope)
        quart = 100.0 / float(my_n_clusters)
        
        quart = (100 + my_n_clusters - 1) / my_n_clusters
        bins = np.percentile(forest.predict(X), np.arange(quart,100,quart))
                          
        # Note we must convert to str to join with slope later
        clusters = pd.DataFrame(index = X.index.astype(str))
        clusters['cluster'] = np.digitize(forest.predict(X), bins)
        print "train cluster cnt: ", np.bincount(clusters.cluster)

        X = X.join(clusters)
        Y = Y.join(clusters)

        best_features_per_cluster = stepwise_best_features_per_cluster(X, Y, all_feature_metadata)
        print "best_features_per_cluster: ", best_features_per_cluster 
        buf = filter_only_selected_features(train_data.set_index("SubjectID"), clusters,                                             best_features_per_cluster)

        s_df = pd.read_csv(StringIO(buf), sep='|', index_col=False, dtype='unicode')
        s_vectorized, _ = vectorize(s_df, all_feature_metadata)
        # if we have a subject missing all selected features, fill him with missing values right before normalizing
        s_vectorized = s_vectorized.join(Y, how = 'right')
        s_vectorized = s_vectorized.drop('ALSFRS_slope', 1)

        s_normalized, _ = normalize(s_vectorized, all_feature_metadata, train_data_means, train_data_std)    
        s_X = s_normalized.join(clusters)
        
        model_per_cluster = get_model_per_cluster(s_X, Y)
        
        return all_feature_metadata, train_data_means, train_data_std, train_data_medians, train_data_mads,                     bins, forest, best_features_per_cluster, model_per_cluster


# In[ ]:

def apply_on_test(test_data, all_feature_metadata, train_data_means, train_data_std, train_data_medians, train_data_mads,
                 clustering_columns, bins, forest, best_features_per_cluster, model_per_cluster):
    
    # Vectorizing
    vectorized, _ = vectorize(test_data, all_feature_metadata)
    cleaned = clean_outliers(vectorized, all_feature_metadata, train_data_medians, train_data_mads, train_data_std)
    normalized, _ = normalize(cleaned, all_feature_metadata, train_data_means, train_data_std)
    
    print "applying on: ", normalized.shape
    
    # Clustering
    
    for_clustering = normalized
    clusters = pd.DataFrame(index = for_clustering.index.astype(str))
    clusters['cluster'] = np.digitize(forest.predict(for_clustering), bins)
    print "applied cluster cnt: ", np.bincount(clusters.cluster)

    X = normalized.join(clusters)
    
    buf = filter_only_selected_features(test_data.set_index("SubjectID"), clusters,                                         best_features_per_cluster)    
    s_df = pd.read_csv(StringIO(buf), sep='|', index_col=False, dtype='unicode')
    s_vectorized, _ = vectorize(s_df, all_feature_metadata)
    s_cleaned = clean_outliers(s_vectorized, all_feature_metadata, train_data_medians, train_data_mads, train_data_std)
    s_normalized, _ = normalize(s_cleaned, all_feature_metadata, train_data_means, train_data_std)    
    input_for_model = s_normalized.join(clusters)    
    
    pred = input_for_model.apply(apply_model, args=[model_per_cluster], axis = 1)
    return input_for_model, pred
    


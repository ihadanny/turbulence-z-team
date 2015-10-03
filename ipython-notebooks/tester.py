
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pickle, cPickle
from datetime import datetime

from vectorizing_funcs import *
from modeling_funcs import *


# In[11]:

test_data = pd.read_csv('../all_forms_validate_leader.txt', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
slope = pd.read_csv('../ALSFRS_slope_validate_leader2.txt', sep = '|', index_col="SubjectID")
slope.index = slope.index.astype(str)

models_folder = "../"

all_feature_metadata = pickle.load( open(models_folder + '/all_feature_metadata.pickle', 'rb') )
train_data_means = pickle.load( open(models_folder + '/all_data_means.pickle', 'rb') )
train_data_std = pickle.load( open(models_folder + '/all_data_std.pickle', 'rb') )
train_data_medians = pickle.load( open(models_folder + '/all_data_medians.pickle', 'rb') )
train_data_mads = pickle.load( open(models_folder + '/all_data_mads.pickle', 'rb') )
clustering_model = cPickle.load( open(models_folder + '/forest_clustering_model.pickle', 'rb') )
best_features_per_cluster = pickle.load( open(models_folder + '/best_features_per_cluster.pickle', 'rb') )
model_per_cluster = pickle.load( open(models_folder + '/model_per_cluster.pickle', 'rb') )

bins = clustering_model["bins"]
forest = clustering_model["model"]

input_for_model, pred = apply_on_test(test_data, all_feature_metadata, 
            train_data_means, train_data_std, train_data_medians, train_data_mads, 
            clustering_columns, bins, forest, best_features_per_cluster, model_per_cluster)

res = pred.join(slope)
good_res = res[~np.isnan(res.ALSFRS_slope)]
print "good_res: ", good_res.shape
test_rmse = np.sqrt(np.mean((good_res.prediction - good_res.ALSFRS_slope) ** 2))
print "RMS Error on test: ", test_rmse
print 'pearson correlation r = %.2f ' % scipy.stats.pearsonr(good_res.prediction, good_res.ALSFRS_slope)[0]


input_for_model.to_csv('../x_results/test_input_for_model.csv',sep='|')
res.to_csv('../x_results/test_prediction.csv',sep='|')




# In[ ]:




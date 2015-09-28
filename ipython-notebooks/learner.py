
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pickle, cPickle
from datetime import datetime

from vectorizing_funcs import *
from modeling_funcs import *


# In[ ]:

df = pd.read_csv('../all_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
slope = pd.read_csv('../all_slope.csv', sep = '|', index_col="SubjectID")
slope.index = slope.index.astype(str)

print "*"*30
tick = datetime.now()

all_feature_metadata, train_data_means, train_data_std, train_data_medians, train_data_mads,              bins, forest, best_features_per_cluster, model_per_cluster = train_it(df, slope, 4)
clustering_model = {"model": forest, "bins": bins}

tock = datetime.now()   
diff = tock - tick 
print "minutes we learned: ", diff.seconds / 60

pickle.dump( all_feature_metadata, open('../all_feature_metadata.pickle', 'wb') )
pickle.dump( train_data_means, open('../all_data_means.pickle', 'wb') )
pickle.dump( train_data_std, open('../all_data_std.pickle', 'wb') )
pickle.dump( train_data_medians, open('../all_data_medians.pickle', 'wb') )
pickle.dump( train_data_mads, open('../all_data_mads.pickle', 'wb') )
pickle.dump( clustering_model, open('../forest_clustering_model.pickle', 'wb') )
pickle.dump( best_features_per_cluster, open("../best_features_per_cluster.pickle", "wb"))
pickle.dump( model_per_cluster, open("../model_per_cluster.pickle", "wb"))

print "pickled all models"
print "*"*30


# In[ ]:




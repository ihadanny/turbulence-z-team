
# coding: utf-8

# ## Builds a model for vectorizing the raw data (apply it once on train and once on test) :
# * pivot from the initial feature_name:feature_value form to a vector
# * handle dummy variables: translate categoric variables into N-1 dummy variables (The model is based on categories in train data)
# * handle time-series variables: reduce them in several hard-coded methods
# * fill missing values with train data means, and normalize to z-scores with train data std
# 

# In[1]:

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from vectorizing_funcs import *


# In[2]:

df = pd.read_csv('../train_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.head()


# # Build metadata: assign features to vectorizing functions
# funcs_to_features arrays define pairs of funcs (can be a list of functions or a single one) and features that should get these functions calculated. Overlapping is allowed.
# 
# There is a list for time-series functions (as described before) and for dummy functions. Both are inverted to feature_to_funcs maps.

# In[3]:

ts_funcs_to_features = [ 
    { 
        "funcs": [ ts_stats, ts_mean_slope, ts_pct_diff ],
        "features": [
            'ALSFRS_Total', 'weight', 'Albumin', 'Creatinine',
            'bp_diastolic', 'bp_systolic', 'pulse', 'respiratory_rate', 'temperature',
        ]
    },
    {
        "funcs": ts_last_value,
        "features": [
            'ALSFRS_Total', 'BMI', 'height', 'Age', 'onset_delta', 'Albumin', 'Creatinine',
        ]
    },
    { 
        "funcs": ts_pct_diff,
        "features": [ 
            'fvc_percent',
        ]
    },
    {
        "funcs": ts_last_boolean,
        "features": [
            'family_ALS_hist',
        ]
    }
]

dummy_funcs_to_features = [ 
    { 
        "funcs": apply_scalar_feature_to_dummies,
        "features": [ 'Gender', 'Race' ]
    }   
]

def invert_func_to_features(ftf, feature_type):
    res = {}
    for ff in ftf:
        funcs = ff['funcs']
        features = ff['features']
        if not type(funcs) is list:
            funcs = [funcs] # a single function
        for func in funcs: 
            for feature in features:
                if feature not in res:
                    res[feature] = {"feature_name": feature, "funcs": set(), 
                                    "feature_type": feature_type, "derived_features": set()}
                res[feature]["funcs"].add(func)
    return res
    
all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))


# ## Learn to_dummies model
# Which kind of categories do we have available in our train data?

# In[4]:

def learn_to_dummies_model(df, all_feature_metadata):
    new_metadata = all_feature_metadata.copy()
    for feature, fv in all_feature_metadata.iteritems():
        if fv["feature_type"] == "dummy":
            for func in fv["funcs"]:
                new_metadata[feature]["derived_features"] = learn_scalar_feature_to_dummies(df, fv)
    return new_metadata

all_feature_metadata = learn_to_dummies_model(df, all_feature_metadata)


# ##Vectorize `train` data 

# In[5]:


vectorized, all_feature_metadata = vectorize(df, all_feature_metadata, debug=True)
vectorized.head()


# ## Filling empty values with means and normalizing
# - NOTE that we have to use the `train` data means and std

# In[6]:

train_data_means = vectorized.mean()
train_data_std = vectorized.std()            
normalized, all_feature_metadata = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
normalized.head()


# In[ ]:




# ## Pickle all metadata we will need to use later when applying vectorizer

# In[7]:

pickle.dump( all_feature_metadata, open('../all_feature_metadata.pickle', 'wb') )
pickle.dump( train_data_means, open('../train_data_means.pickle', 'wb') )
pickle.dump( train_data_std, open('../train_data_std.pickle', 'wb') )


# ## Apply model on `train`,  `test` 
# 

# In[8]:


for t in ["train", "test"]:
    df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
    vectorized, _ = vectorize(df, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
    print t, normalized.shape
    normalized.to_csv('../' + t + '_data_vectorized.csv' ,sep='|')

normalized.head()


# In[ ]:




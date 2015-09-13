
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


all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))


# ## Learn to_dummies model
# Which kind of categories do we have available in our train data?

# In[4]:

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




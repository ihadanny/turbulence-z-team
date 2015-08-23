
# coding: utf-8

# In[1]:

## Used for vectorizing the raw data (run it once on train and once on test) :
## Pivoting it from the initial feature_name:feature_value form to a vector
## scalar_feature_to_dummies - Translating categoric variables into N-1 dummy variables
## timeseries_feature_slope_reduced - mean, std for time series variables (have multiple measurements in different times)
## timeseries_feature_last_value - take last value in time series
## Filling empty values with means - NOTE that these have to be the train data means


# In[2]:

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict


# In[3]:

def parse_feature_delta(fd):
    if type(fd) is float: return fd
    first_value = fd.split(';')[0]
    try:
        return float(first_value)
    except:
        return None


# In[4]:

df = pd.read_csv('../train_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.loc[:,'feature_delta'] = df.feature_delta.apply(parse_feature_delta)
df = df[df.feature_delta < 92]

df.head()


# In[5]:

print "\n".join( (df.form_name + " - " + df.feature_name).unique() )


# In[ ]:




# # Define all kind of vectorization and aggregation functions

# ## Global functions
# Should receive (df, feature_name) and return a DataFrame with SubjectID as an index and columns for features

# ### Scalar -> Dummies

# In[6]:

def scalar_feature_to_dummies(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    my_slice_pivot = pd.pivot_table(my_slice, values = ['feature_value'], index = ['SubjectID'], 
                                columns = ['feature_name'], aggfunc = lambda x:x)
    dum = pd.get_dummies(my_slice_pivot['feature_value'][feature_name])
    return dum


# ## Time Series functions
# Are invoked per SubjectID and with the valid timeframe data only (<92 days). Should receive a DataFrame with 'feature_value', and 'feature_delta' and return a dict from col_suffix (e.g. "last", "mean", ...) to the value

# ### Timeseries -> Slope, %diff, stats

# In[7]:

def ts_pct_diff(ts_data):
    if len(ts_data) < 2:
        return None
    
    ts_data_sorted = ts_data.sort('feature_delta')
    values = ts_data_sorted.feature_value.astype('float')
    time_values = ts_data_sorted.feature_delta.astype('float')

    time_diff = time_values.iloc[-1] - time_values.iloc[0]
    val = ( values.iloc[-1] - values.iloc[0] ) / ( values.iloc[0] * time_diff)
    if val == float('inf'):
        return None
    
    return { "pct_diff": val }
    
def ts_stats(ts_data):
    if len(ts_data) < 1:
        return None
    
    values = ts_data.feature_value.astype('float')
    return { "mean": values.mean(), "std": values.std(), "median": values.median() }
    
def ts_mean_slope(ts_data):
    if len(ts_data) < 2:
        return None
    
    ts_data_sorted = ts_data.sort('feature_delta') 
    ts_data_sorted.feature_value = ts_data_sorted.feature_value.astype('float')
    first, others = ts_data_sorted.iloc[0], ts_data_sorted.iloc[1:]
    slopes = [ ( x[1].feature_value - first.feature_value) / ( x[1].feature_delta - first.feature_delta ) for x in others.iterrows() ]
    slopes = [ x for x in slopes if x!=float('inf') ]
    return { "mean_slope": np.mean(slopes) }


# ## Timeseries -> last value

# In[8]:

def ts_last_value(ts_data):
    if len(ts_data) < 1:
        return None
    
    ts_data_sorted = ts_data.sort('feature_delta') 
    return { "last": ts_data_sorted.feature_value.astype('float').iloc[-1] }


# ## Special Treatment

# In[9]:

def last_boolean(ts_data):
    if len(ts_data) < 1:
        return None
    val_str = str(ts_data.feature_value.iloc[-1]).lower()
    if val_str == 'y' or val_str == 'true':
        val = 1
    else:
        val = 0
    return { "last": val }
    


# # Assign features to functions
# funcs_to_features arrays define pairs of funcs (can be a list of functions or a single one) and features that should get these functions calculated. Overlapping is allowed.
# 
# There is a list for time-series functions (as described before) and for global function (like scalar->dummies). Both are inverted to feature_to_funcs maps.

# In[10]:

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
        "funcs": last_boolean,
        "features": [
            'family_ALS_hist',
        ]
    }
]

global_funcs_to_features = [ 
    { 
        "funcs": scalar_feature_to_dummies,
        "features": [ 'Gender', 'Race' ]
    }   
]

def invert_func_to_features(ftf):
    res = defaultdict(set)
    for ff in ftf:
        funcs = ff['funcs']
        features = ff['features']
        if not type(funcs) is list:
            funcs = [funcs] # a single function
        for func in funcs: 
            for feature in features:
                res[feature].add(func)
    return res
    
ts_feature_to_funcs = invert_func_to_features(ts_funcs_to_features)
global_feature_to_funcs = invert_func_to_features(global_funcs_to_features)

all_feature_to_funcs = ts_feature_to_funcs.copy()
all_feature_to_funcs.update(global_feature_to_funcs)


# ## Calculate all features

# In[18]:

def to_series(f):
    def foo(x):
        res = f(x)
        if res is None: 
            return None
        else:
            return pd.Series(f(x))
        
    return foo


def vectorize(df, ts_feature_to_funcs, global_feature_to_funcs):
    vectorized = pd.DataFrame(index=df.SubjectID.unique())
    feature_groups = defaultdict(set)
    
    # Global functions: receive (df,feature), return DataFrame with SubjectID as index and columns for features
    for feature, funcs in global_feature_to_funcs.iteritems():
        for func in funcs: 
            res = func(df, feature)
            vectorized = pd.merge(vectorized, res, how='left', right_index=True, left_index=True)
            for f in res.columns:
                feature_groups[feature].add(f)
    
    
    # Time Series functions: receive time-series data, return a dict from feature_suffix ("mean", "median", "last", ...) to value
    # Those are being run on a specific SubjectID and within the allowed timeframe.
    pointintime_data = df[df.feature_delta < 92]
    pointintime_data = pointintime_data.drop_duplicates(subset = ['SubjectID', 'feature_name' ,'feature_delta'], take_last=True)
    
    for feature, funcs in ts_feature_to_funcs.iteritems():
        feature_ts_data = pointintime_data[pointintime_data.feature_name == feature]
        for func in funcs: 
            res = pd.DataFrame(feature_ts_data.groupby('SubjectID').apply(to_series(func)))
            res.columns = [ feature + "_" + str(col_suffix) for col_suffix in res.columns ]
            vectorized = pd.merge(vectorized, res, how='left', right_index=True, left_index=True)  
            for f in res.columns:
                feature_groups[feature].add(f)

    return vectorized, feature_groups


# In[19]:

vectorized, feature_groups = vectorize(df, ts_feature_to_funcs, global_feature_to_funcs)
vectorized.head()


# In[ ]:




# ## Filling empty values with means
# - NOTE that these have to be the train data means

# In[13]:

train_data_means = vectorized.mean()
vectorized = vectorized.fillna(train_data_means)
vectorized.head()


# # Calculate ZScore for all columns

# In[14]:

def calc_all_zscore(vectorized):
    for col in vectorized.columns:
        try:
            col_zscore = col + '_zscore'
            data = vectorized[col].astype('float')
            vectorized[col_zscore] = (data - data.mean())/data.std(ddof=0)
        except:
            pass
            


# In[ ]:




# ## Run everything on `test` and `train`

# In[15]:

train_vectorized = vectorize(df, ts_feature_to_funcs, global_feature_to_funcs)
train_data_means = train_vectorized.mean()

for t in ["train", "test"]:
    df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
    df.loc[:,'feature_delta'] = df.feature_delta.apply(parse_feature_delta)
    df = df[df.feature_delta < 92]

    vectorized, feature_groups = vectorize(df, ts_feature_to_funcs, global_feature_to_funcs)
    vectorized = vectorized.fillna(train_data_means)
    calc_all_zscore(vectorized)
  
    vectorized.index.name='SubjectID'
    print t, vectorized.shape
    vectorized.to_csv('../' + t + '_data_vectorized.csv' ,sep='|')
    

pickle.dump( feature_groups, open('../feature_groups.pickle', 'wb') )


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




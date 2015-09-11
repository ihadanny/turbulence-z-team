
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

df = pd.read_csv('../train_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.head()


# In[ ]:




# # Define all kind of vectorization and aggregation functions

# ## Global functions
# Should receive (df, feature_name) and return a DataFrame with SubjectID as an index and columns for features

# ### Scalar -> Dummies

# In[4]:

def scalar_feature_to_dummies_core(df, feature_metadata):
    my_slice = df[df.feature_name == feature_metadata["feature_name"]]
    my_slice_pivot = pd.pivot_table(my_slice, values = ['feature_value'], index = ['SubjectID'], 
                                columns = ['feature_name'], aggfunc = lambda x:x)
    dum = pd.get_dummies(my_slice_pivot['feature_value'][feature_metadata["feature_name"]])
    return dum

def learn_scalar_feature_to_dummies(df, feature_metadata):
    dum = scalar_feature_to_dummies_core(df, feature_metadata)
    return dum.columns

def apply_scalar_feature_to_dummies(df, feature_metadata):
    dum = scalar_feature_to_dummies_core(df, feature_metadata)
    dum.reindex(columns = feature_metadata["derived_features"], fill_value=0)   
    return dum


# ## Time Series functions
# Are invoked per SubjectID and with the valid timeframe data only (<92 days). Should receive a DataFrame with 'feature_value', and 'feature_delta' and return a dict from col_suffix (e.g. "last", "mean", ...) to the value
# 
# NOTE: here theres no learned model - we apply the same hard-coded treatment 

# ### Timeseries -> Slope, %diff, stats

# In[5]:

def ts_pct_diff(ts_data, feature_metadata):
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
    
def ts_stats(ts_data, feature_metadata):
    if len(ts_data) < 1:
        return None
    
    values = ts_data.feature_value.astype('float')
    return { "mean": values.mean(), "std": values.std(), "median": values.median() }
    
def ts_mean_slope(ts_data, feature_metadata):
    if len(ts_data) < 2:
        return None
    
    ts_data_sorted = ts_data.sort('feature_delta') 
    ts_data_sorted.feature_value = ts_data_sorted.feature_value.astype('float')
    first, others = ts_data_sorted.iloc[0], ts_data_sorted.iloc[1:]
    slopes = [ ( x[1].feature_value - first.feature_value) / ( x[1].feature_delta - first.feature_delta ) for x in others.iterrows() ]
    slopes = [ x for x in slopes if x!=float('inf') ]
    return { "mean_slope": np.mean(slopes) }


# ## Timeseries -> last value

# In[6]:

def ts_last_value(ts_data, feature_metadata):
    if len(ts_data) < 1:
        return None
    
    ts_data_sorted = ts_data.sort('feature_delta') 
    return { "last": ts_data_sorted.feature_value.astype('float').iloc[-1] }


# In[7]:

def ts_last_boolean(ts_data, feature_metadata):
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

# In[8]:

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
                    res[feature] = {"feature_name": feature, "funcs": set(), "feature_type": feature_type}
                res[feature]["funcs"].add(func)
    return res
    
ts_feature_to_funcs = invert_func_to_features(ts_funcs_to_features, "ts")
dummy_feature_to_funcs = invert_func_to_features(dummy_funcs_to_features, "dummy")

all_feature_metadata = ts_feature_to_funcs.copy()
all_feature_metadata.update(dummy_feature_to_funcs)


# ## Learn to_dummies model
# Which kind of categories do we have available in our train data?

# In[9]:

def learn_to_dummies_model(df, all_feature_metadata):
    new_metadata = all_feature_metadata.copy()
    for feature, fv in all_feature_metadata.iteritems():
        if fv["feature_type"] == "dummy":
            for func in fv["funcs"]:
                new_metadata[feature]["derived_features"] = learn_scalar_feature_to_dummies(df, fv)
    return new_metadata

all_feature_metadata = learn_to_dummies_model(df, all_feature_metadata)


# In[ ]:




# In[10]:

def to_series(f):
    def foo(x, args):
        res = f(x, args)
        if res is None: 
            return None
        else:
            return pd.Series(res)
    return foo

def parse_feature_delta(fd):
    """ parse feature_delta which can be given in strange forms, such as '54;59' """
    if type(fd) is float or type(fd) is np.float64: return fd
    first_value = fd.split(';')[0]
    try:
        return float(first_value)
    except:
        return None


def vectorize(df, all_feature_metadata):
    vectorized = pd.DataFrame(index=df.SubjectID.unique())
    df.loc[:,'feature_delta'] = df.feature_delta.apply(parse_feature_delta)
    pointintime_data = df[df.feature_delta < 92]
    pointintime_data = pointintime_data.drop_duplicates(subset = ['SubjectID', 'feature_name' ,'feature_delta'], take_last=True)
    new_metadata = all_feature_metadata.copy()
    for feature, fm in all_feature_metadata.iteritems():
        if fm["feature_type"] == "dummy":
            for func in fm["funcs"]:
                res = func(df, fm)
        elif fm["feature_type"] == "ts":    
            # Time Series functions: receive time-series data, return a dict from feature_suffix ("mean", "median", "last", ...) to value
            # Those are being run on a specific SubjectID and within the allowed timeframe.
            feature_ts_data = pointintime_data[pointintime_data.feature_name == feature]
            for func in fm["funcs"]: 
                res = pd.DataFrame(feature_ts_data.groupby('SubjectID').apply(to_series(func), args=fm))
                res.columns = [ feature + "_" + str(col_suffix) for col_suffix in res.columns ]
        else:
            raise Exception("unknown feature type: " + fv["feature_type"])
        vectorized = pd.merge(vectorized, res, how='left', right_index=True, left_index=True)
        new_metadata[feature]["derived_features"] = res.columns

    vectorized.index.name='SubjectID'
    return vectorized, new_metadata


# In[11]:

vectorized, all_feature_metadata = vectorize(df, all_feature_metadata)
vectorized.head()


# ## Filling empty values with means and normalizing
# - NOTE that we have to use the `train` data means and std

# In[12]:

train_data_means = vectorized.mean()
train_data_std = vectorized.std()

def normalize(vectorized, all_feature_metadata, train_data_means, train_data_std):
    normalized = vectorized.fillna(train_data_means)
    for feature, fm in all_feature_metadata.iteritems():
        for col in fm["derived_features"]:
            data = normalized[col].astype('float')
            normalized.loc[:, col] = (data - train_data_means[col])/train_data_std[col]
    return normalized, all_feature_metadata
            
normalized, all_feature_metadata = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
print normalized.columns
normalized.head()


# In[ ]:




# ## Pickle everything we need to use later when vectorizing

# In[15]:

pickle.dump( all_feature_metadata, open('../all_feature_metadata.pickle', 'wb') )
pickle.dump( train_data_means, open('../train_data_means.pickle', 'wb') )
pickle.dump( train_data_std, open('../train_data_std.pickle', 'wb') )


# ## Apply model on `train`, and then apply model on `test` subject by subject, as thats the required mod-op in production

# In[16]:


t = "train"
df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
vectorized, _ = vectorize(df, all_feature_metadata)
normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)
print t, normalized.shape
normalized.to_csv('../' + t + '_data_vectorized.csv' ,sep='|')

t = "test"
df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
for subj in df.SubjectID.unique():
    df_subj = df[df.SubjectID == subj]
    vectorized, _ = vectorize(df_subj, all_feature_metadata)
    normalized, _ = normalize(vectorized, all_feature_metadata, train_data_means, train_data_std)

print t, normalized.shape
normalized.to_csv('../' + t + '_data_vectorized.csv' ,sep='|')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




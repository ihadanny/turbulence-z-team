
# coding: utf-8

# ## Methods used for vectorizing the raw data 
# * Used several times across our flow

# In[28]:

import pandas as pd
import copy
import numpy as np
import pickle
from collections import defaultdict


# In[ ]:




# # Define all kind of vectorization and aggregation functions

# ## Global functions
# Should receive (df, feature_name) and return a DataFrame with SubjectID as an index and columns for features

# ### Scalar -> Dummies

# In[29]:

def scalar_feature_to_dummies_core(df, feature_metadata):
    my_slice = df[df.feature_name == feature_metadata["feature_name"]]
    if my_slice.shape[0] == 0:
        return pd.DataFrame()
    my_slice_pivot = pd.pivot_table(my_slice, values = ['feature_value'], index = ['SubjectID'], 
                                columns = ['feature_name'], aggfunc = lambda x:x)
    dum = pd.get_dummies(my_slice_pivot['feature_value'][feature_metadata["feature_name"]])
    return dum

def learn_scalar_feature_to_dummies(df, feature_metadata):
    dum = scalar_feature_to_dummies_core(df, feature_metadata)
    return dum.columns

def apply_scalar_feature_to_dummies(df, feature_metadata):
    dum = scalar_feature_to_dummies_core(df, feature_metadata)
    return dum.reindex(columns = feature_metadata["derived_features"], fill_value=0)   


# ## Time Series functions
# Are invoked per SubjectID and with the valid timeframe data only (<92 days). Should receive a DataFrame with 'feature_value', and 'feature_delta' and return a dict from col_suffix (e.g. "last", "mean", ...) to the value
# 
# NOTE: here theres no learned model - we apply the same hard-coded treatment 

# ### Timeseries -> Slope, %diff, stats

# In[30]:

def ts_pct_diff(ts_data, feature_metadata):
    if len(ts_data) < 2:
        return None
    try:
        ts_data_sorted = ts_data.sort('feature_delta')
        values = ts_data_sorted.feature_value.astype('float')
        time_values = ts_data_sorted.feature_delta.astype('float')

        time_diff = time_values.iloc[-1] - time_values.iloc[0]
        val = ( values.iloc[-1] - values.iloc[0] ) / ( values.iloc[0] * time_diff)
        if val == float('inf'):
            return None

        return { "pct_diff": val }
    except Exception:
        return None
        
    
def ts_stats(ts_data, feature_metadata):
    if len(ts_data) < 1:
        return None
    try:
        values = ts_data.feature_value.astype('float')
        return { "mean": values.mean() }
        #return { "mean": values.mean(), "std": values.std(), "median": values.median() }
    except Exception:
        return None

def ts_mean_slope(ts_data, feature_metadata):
    if len(ts_data) < 2:
        return None
    try:
        ts_data_sorted = ts_data.sort('feature_delta') 
        ts_data_sorted.feature_value = ts_data_sorted.feature_value.astype('float')
        first, others = ts_data_sorted.iloc[0], ts_data_sorted.iloc[1:]
        slopes = [ ( x[1].feature_value - first.feature_value) / ( x[1].feature_delta - first.feature_delta ) for x in others.iterrows() ]
        slopes = [ x for x in slopes if x!=float('inf') ]
        return { "mean_slope": np.mean(slopes) }
    except Exception:
        return None


# ## Timeseries -> last value

# In[31]:

def ts_last_value(ts_data, feature_metadata):
    if len(ts_data) < 1:
        return None
    try:
        ts_data_sorted = ts_data.sort('feature_delta') 
        return { "last": ts_data_sorted.feature_value.astype('float').iloc[-1] }
    except Exception:
        return None


# In[32]:

def ts_last_boolean(ts_data, feature_metadata):
    if len(ts_data) < 1:
        return None
    try:
        val_str = str(ts_data.feature_value.iloc[-1]).lower()
        if val_str == 'y' or val_str == 'true' or val_str == 'yes':
            val = 1
        else:
            val = 0
        return { "last": val }
    except Exception:
        return None


# ## Metadata
# Static part of our metadata - which feature maps to which vectorizing func?

# In[34]:

ts_funcs_to_features = [ 
    { 
        "funcs": [ ts_stats, ts_mean_slope, ts_pct_diff, ts_last_value ],
        "features": [
            'ALSFRS_Total', 'weight', 'Albumin', 'Creatinine',
            'bp_diastolic', 'bp_systolic', 'pulse', 'respiratory_rate', 'temperature',
            'mouth', 'respiratory', 'hands', 'fvc_percent'
        ]
    },
    {
        "funcs": ts_last_value,
        # following features appear only once, so we just take their last values 
        "features": [
            'BMI', 'Age', 'onset_delta',
        ]
    },
    {
        "funcs": ts_last_boolean,
        "features": [
            'family_ALS_hist', 'if_use_Riluzole',
        ]
    }
]

# Adding all lab tests that more than 1800 subjects took more than twice each
# TODO: this shouldnt be static, we should do it per fold for the train data
def add_frequent_lab_tests_to_ts_features(df, ts_funcs_to_features): 
    metadata =  copy.deepcopy(ts_funcs_to_features)
    lab = df[df.form_name == 'Lab Test']
    lab_numeric = lab[~np.isnan(lab.feature_value.convert_objects(convert_numeric=True))]
    by_feature = lab_numeric.groupby(["feature_name", "SubjectID"])
    features_with_multiple_visits = by_feature.filter(lambda x: len(x)>2)
    distinct_subjects = features_with_multiple_visits.groupby("feature_name").SubjectID.nunique()
    lab_having_enough_samples = distinct_subjects[distinct_subjects > df.SubjectID.nunique() * 0.75]

    for f in lab_having_enough_samples.index:
        metadata[0]["features"].append(f)
    return metadata

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


# In[36]:

def learn_to_dummies_model(df, all_feature_metadata):
    new_metadata = all_feature_metadata.copy()
    for feature, fv in all_feature_metadata.iteritems():
        if fv["feature_type"] == "dummy":
            for func in fv["funcs"]:
                new_metadata[feature]["derived_features"] = learn_scalar_feature_to_dummies(df, fv)
    return new_metadata


# ## Helper functions

# In[37]:

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


# ## Vectorize

# In[38]:


def vectorize(df, all_feature_metadata, debug=False):
    vectorized = pd.DataFrame(index=df.SubjectID.unique().astype(str)) # SubjectID is always str for later joins
    df.loc[:,'feature_delta'] = df.feature_delta.apply(parse_feature_delta)
    pointintime_data = df[df.feature_delta < 92]
    pointintime_data = pointintime_data.drop_duplicates(subset = ['SubjectID', 'feature_name' ,'feature_delta'], take_last=True)
    new_metadata = all_feature_metadata.copy()
    for feature, fm in all_feature_metadata.iteritems():
        feature_ts_data = pointintime_data[pointintime_data.feature_name == feature]
        for func in fm["funcs"]:
            if fm["feature_type"] == "dummy":
                res = func(df, fm)
            elif fm["feature_type"] == "ts":    
                res = pd.DataFrame(feature_ts_data.groupby('SubjectID').apply(to_series(func), args=fm))
                res.columns = [ feature + "_" + str(col_suffix) for col_suffix in res.columns ]
                for col in res.columns:
                    new_metadata[feature]["derived_features"].add(col)
            else:
                raise Exception("unknown feature type: " + fv["feature_type"])
            vectorized = pd.merge(vectorized, res, how='left', right_index=True, left_index=True)
        if debug:
            print feature

    vectorized.index.name='SubjectID'
    return vectorized, new_metadata


# ## Normalize

# In[39]:

def normalize(vectorized, all_feature_metadata, train_data_means, train_data_std):
    vectorized = vectorized.reindex(columns=train_data_means.keys())
    normalized = vectorized.fillna(train_data_means)
    for feature, fm in all_feature_metadata.iteritems():
        for col in fm["derived_features"]:
            data = normalized[col].astype('float')
            normalized.loc[:, col] = (data - train_data_means[col])/train_data_std[col]
    return normalized, all_feature_metadata


# In[ ]:




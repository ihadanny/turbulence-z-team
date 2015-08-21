
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
from collections import defaultdict


# In[3]:

df = pd.read_csv('../train_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.describe()


# In[4]:

set(df.feature_name)


# In[5]:

func_per_feature = defaultdict(set)

vectorized = pd.DataFrame(index=df['SubjectID'].unique())
print vectorized.shape


# In[ ]:




# In[6]:

def scalar_feature_to_dummies(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    my_slice_pivot = pd.pivot_table(my_slice, values = ['feature_value'], index = ['SubjectID'], 
                                columns = ['feature_name'], aggfunc = lambda x:x)
    dum = pd.get_dummies(my_slice_pivot['feature_value'][feature_name])
    return dum

for feature_name in ['Gender', 'Race']:
    func_per_feature[feature_name].add(scalar_feature_to_dummies)
    vectorized = pd.merge(vectorized, scalar_feature_to_dummies(df, feature_name), how = 'left',
                          right_index=True, left_index=True)  

vectorized.head()


# In[9]:

### Calculating slope - the diffs between each measurement and the first measurement (0 day) 
def calc_slope(row) :
    time_delta =  (float(row['feature_delta_int_y']) - float(row['feature_delta_int_x']))
    return (row['feature_value_float_y'] - row['feature_value_float_x'])/time_delta

def timeseries_feature_to_slope(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    # There were duplicate measurements of timeseries features with the same feature_delta :(
    my_slice = my_slice.drop_duplicates(subset = ['SubjectID', 'feature_delta'], take_last=True)
    my_slice.loc[:, 'feature_value_float'] = my_slice['feature_value'].astype(float)
    my_slice.loc[:, 'feature_delta_int'] = my_slice['feature_delta'].astype(float)
    my_slice_other_visits = my_slice[(my_slice.feature_delta_int > 0) & (my_slice.feature_delta_int < 92)]
    my_slice_first_visit = my_slice[my_slice.feature_delta_int == 0]
    my_slice_j = pd.merge(my_slice_first_visit, my_slice_other_visits, on=['SubjectID','feature_name']) 
    my_slice_j.loc[:, 'feature_value_slope'] = my_slice_j.apply(calc_slope, axis=1)
    return my_slice_j

def timeseries_feature_slope_reduced(df, feature_name):
    res = pd.DataFrame(index=df['SubjectID'].unique())
    for func in ['mean', 'std']:
        slope_series = timeseries_feature_to_slope(df, feature_name)
        
        slope_pivot = pd.pivot_table(slope_series, values = ['feature_value_slope'], index = ['SubjectID'], 
                                     columns = ['feature_name'], aggfunc = func)
        slope_pivot = slope_pivot['feature_value_slope']
        slope_pivot.columns = [feature_name + "_slope_" + func]
        res = pd.merge(res, slope_pivot, how='left', right_index=True, left_index=True)          
    
    return res

def calc_diff_pct(group):
    if len(group) < 2:
        return None
    
    group_sorted = group.sort('feature_delta')
    values = group_sorted.feature_value.astype('float')
    time_values = group_sorted.feature_delta.astype('float')

    time_diff = time_values.iloc[-1] - time_values.iloc[0]
    return ( values.iloc[-1] - values.iloc[0] ) / ( values.iloc[0] * time_diff)
    
def timeseries_feature_pct_diff(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    ret = pd.DataFrame(my_slice.groupby('SubjectID').apply(calc_diff_pct))
    ret.columns = [ feature_name + "_pct_diff" ]
    return ret

def timeseries_feature_stats(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    my_slice.loc[:,'feature_value'] = my_slice.feature_value.astype('float')
    ret = pd.DataFrame(my_slice.groupby('SubjectID').feature_value.agg([np.mean, np.median, np.std]))
    ret.columns = [ feature_name + "_" + func_name for func_name in [ "mean", "median", "std"] ]
    return ret


def timeseries_feature(df, feature_name):
    res = pd.DataFrame(index=df['SubjectID'].unique())
    
    res = pd.merge(res, timeseries_feature_slope_reduced(df, feature_name), how='left',
                          right_index=True, left_index=True )
    res = pd.merge(res, timeseries_feature_pct_diff(df, feature_name), how='left',
                          right_index=True, left_index=True ) 
    res = pd.merge(res, timeseries_feature_stats(df, feature_name), how='left',
                          right_index=True, left_index=True ) 
    return res

for feature_name in [
    'ALSFRS_Total', 'weight', 
    'bp_diastolic', 'bp_systolic', 'pulse', 'respiratory_rate', 'temperature' ]:
    func_per_feature[feature_name].add(timeseries_feature)
    vectorized = pd.merge(vectorized, timeseries_feature(df, feature_name), how='left',
                          right_index=True, left_index=True)  
    
vectorized.head()



# In[ ]:




# In[10]:

def timeseries_feature_last_value(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    ret = my_slice.groupby('SubjectID').last().loc[:, ['feature_value']].astype(float)
    ret.columns = [feature_name + "_last"]
    return ret

for feature_name in [
    'ALSFRS_Total', 'BMI', 'height', 'Age']:
    func_per_feature[feature_name].add(timeseries_feature_last_value)
    vectorized = pd.merge(vectorized, timeseries_feature_last_value(df, feature_name), how='left',
                          right_index=True, left_index=True)  
vectorized.head()


# ## Other functions

# In[ ]:




# In[ ]:




# In[ ]:




# ## Filling empty values with means - NOTE that these have to be the train data means

# In[11]:

train_data_means = vectorized.mean()
vectorized = vectorized.fillna(train_data_means)
vectorized.head()


# In[12]:

# Calcualte ZScore for all columns
def calc_all_zscore(vectorized):
    for col in vectorized.columns:
        col_zscore = col + '_zscore'
        vectorized[col_zscore] = (vectorized[col] - vectorized[col].mean())/vectorized[col].std(ddof=0)


# In[13]:

def parse_feature_delta(fd):
    if type(fd) is float: return fd
    first_value = fd.split(';')[0]
    try:
        return float(first_value)
    except:
        return None


# ## Run everything on `test` and `train`

# In[205]:

for t in ["train", "test"]:
    df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
    df.loc[:,'feature_delta'] = df.feature_delta.apply(parse_feature_delta)
    df = df[df.feature_delta < 92]

    vectorized = pd.DataFrame(index=df['SubjectID'].unique())
    for feature_name, funcs in func_per_feature.iteritems():
        for func in funcs:
            vectorized = pd.merge(vectorized, func(df, feature_name), how = 'left',
                      right_index=True, left_index=True)  
    final_data = vectorized.fillna(train_data_means)
    calc_all_zscore(final_data)
    
    final_data.index.name='SubjectID'
    print t, final_data.shape
    final_data.to_csv('../' + t + '_data_vectorized.csv' ,sep='|')


# In[15]:

func_per_feature


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




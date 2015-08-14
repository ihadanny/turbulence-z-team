
# coding: utf-8

# In[54]:

## Used for vectorizing the raw data (run it once on train and once on test) :
## Pivoting it from the initial feature_name:feature_value form to a vector
## scalar_feature_to_dummies - Translating categoric variables into N-1 dummy variables
## timeseries_feature_slope_reduced - mean, std for time series variables (have multiple measurements in different times)
## timeseries_feature_last_value - take last value in time series
## Filling empty values with means - NOTE that these have to be the train data means


# In[55]:

import pandas as pd
import numpy as np


# In[56]:

df = pd.read_csv('../train_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.describe()


# In[57]:

interesting = df[(df.form_name == 'Demographic') | (df.form_name == 'Vitals')]
print interesting['feature_name'].unique()
func_per_feature = {}
vectorized = pd.DataFrame(index=df['SubjectID'].unique())
print vectorized.shape


# In[58]:

def scalar_feature_to_dummies(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    my_slice_pivot = pd.pivot_table(my_slice, values = ['feature_value'], index = ['SubjectID'], 
                                columns = ['feature_name'], aggfunc = lambda x:x)
    dum = pd.get_dummies(my_slice_pivot['feature_value'][feature_name])
    return dum

for feature_name in ['Gender', 'Race']:
    func_per_feature[feature_name] = scalar_feature_to_dummies
    vectorized = pd.merge(vectorized, func_per_feature[feature_name](df, feature_name), how = 'left',
                          right_index=True, left_index=True)  

vectorized.head()


# In[59]:

### Calculating slope - the diffs between each measurement and the first measurement (0 day) 
def calc_slope(row) :
    time_delta =  (float(row['feature_delta_int_y']) - float(row['feature_delta_int_x']))
    return (row['feature_value_float_y'] - row['feature_value_float_x'])/time_delta

def timeseries_feature_to_slope(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    # There were duplicate measurements of timeseries features with the same feature_delta :(
    my_slice = my_slice.drop_duplicates(subset = ['SubjectID', 'feature_delta'], take_last=True)
    my_slice.loc[:, 'feature_value_float'] = my_slice['feature_value'].astype(float)
    my_slice.loc[:, 'feature_delta_int'] = my_slice['feature_delta'].astype(int)
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
        slope_pivot.columns = [feature_name + "_" + func]
        res = pd.merge(res, slope_pivot, right_index=True, left_index=True)          
    return res

for feature_name in ['bp_diastolic', 'bp_systolic', 'pulse', 'respiratory_rate', 'temperature', 'weight']:
    func_per_feature[feature_name] = timeseries_feature_slope_reduced
    vectorized = pd.merge(vectorized, func_per_feature[feature_name](df, feature_name), how='left',
                          right_index=True, left_index=True)  
    
vectorized.head()


# In[60]:

def timeseries_feature_last_value(df, feature_name):
    my_slice = df[df.feature_name == feature_name]
    ret = my_slice.groupby('SubjectID').last().loc[:, ['feature_value']].astype(float)
    ret.columns = [feature_name + "_last"]
    return ret

for feature_name in ['BMI', 'height']:
    func_per_feature[feature_name] = timeseries_feature_last_value
    vectorized = pd.merge(vectorized, func_per_feature[feature_name](df, feature_name), how='left',
                          right_index=True, left_index=True)  
vectorized.head()


# In[61]:

## Filling empty values with means - NOTE that these have to be the train data means
train_data_means = vectorized.mean()
vectorized = vectorized.fillna(train_data_means)
vectorized.head()


# In[63]:

for t in ["train", "test"]:
    df = pd.read_csv('../' + t + '_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
    vectorized = pd.DataFrame(index=df['SubjectID'].unique())
    for feature_name, func in func_per_feature.iteritems():
        vectorized = pd.merge(vectorized, func_per_feature[feature_name](df, feature_name), how = 'left',
                      right_index=True, left_index=True)  
    final_data = vectorized.fillna(train_data_means)
    final_data.index.name='SubjectID'
    print t, final_data.shape
    final_data.to_csv('../' + t + '_data_vectorized.csv' ,sep='|')


# In[ ]:




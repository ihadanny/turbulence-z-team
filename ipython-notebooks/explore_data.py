
# coding: utf-8

# In[18]:

import pandas as pd
import numpy as np
from IPython.display import display

from  vectorizing_funcs import *
df = pd.read_csv('../all_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.head()


# In[2]:

feature_names = df[["form_name", "feature_name"]].drop_duplicates()
feature_names.to_csv('../feature_names.csv', sep='|', index=False)


# In[3]:

feature_values = df[["form_name", "feature_name", "feature_value"]].drop_duplicates()
feature_values = feature_values[np.isnan(feature_values.feature_value.convert_objects(convert_numeric=True))]
feature_values.to_csv('../feature_values.csv', sep='|', index=False)


# In[5]:

feature_values = df[df.form_name == 'Lab Test']
feature_values = feature_values[~np.isnan(feature_values.feature_value.convert_objects(convert_numeric=True))]
by_subject = feature_values.groupby(["feature_name", "SubjectID"])
features_with_multiple_visits = by_subject.filter(lambda x: len(x)>2)
by_subject = features_with_multiple_visits.groupby("feature_name").SubjectID.nunique()
by_subject.sort(ascending=False)
by_subject[:30]


# In[100]:

all_feature_metadata = invert_func_to_features(ts_funcs_to_features, "ts")
all_feature_metadata.update(invert_func_to_features(dummy_funcs_to_features, "dummy"))
all_feature_metadata = learn_to_dummies_model(df, all_feature_metadata)
vectorized, all_feature_metadata = vectorize(df, all_feature_metadata, debug=True)


# In[101]:

vectorized.describe().transpose()


# In[30]:

slope = pd.read_csv('../all_slope.csv', sep = '|', index_col=0)
slope.index = slope.index.astype(str)

max_date = df[df.feature_name == 'ALSFRS_Total'][['SubjectID','feature_delta']]
max_date.loc[:, 'feature_delta'] = max_date.feature_delta.astype(int)
max_date = max_date.groupby('SubjectID').max()
print max_date.shape, slope.shape
j = slope.join(max_date)
print j.shape
j[j.feature_delta < 365].shape


# In[33]:

df[df.feature_name == 'onset_site'].feature_value.unique()


# In[ ]:




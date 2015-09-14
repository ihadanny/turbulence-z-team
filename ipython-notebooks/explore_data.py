
# coding: utf-8

# In[53]:

import pandas as pd
import numpy as np
from  vectorizing_funcs import *
df = pd.read_csv('../all_data.csv', sep = '|', error_bad_lines=False, index_col=False, dtype='unicode')
df.head()


# In[29]:

feature_names = df[["form_name", "feature_name"]].drop_duplicates()
feature_names.to_csv('../feature_names.csv', sep='|', index=False)


# In[30]:

feature_values = df[["form_name", "feature_name", "feature_value"]].drop_duplicates()
feature_values = feature_values[np.isnan(feature_values.feature_value.convert_objects(convert_numeric=True))]
feature_values.to_csv('../feature_values.csv', sep='|', index=False)


# In[91]:

feature_values = df[df.form_name == 'Lab Test']
feature_values = feature_values[~np.isnan(feature_values.feature_value.convert_objects(convert_numeric=True))]
by_subject = feature_values.groupby("feature_name").SubjectID.nunique()
by_subject.sort(ascending=False)
by_subject[:40]


# In[99]:

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


# In[ ]:




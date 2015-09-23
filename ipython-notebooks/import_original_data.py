
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from IPython.display import display


# In[2]:

df = pd.read_csv('../../all_forms_PROACT.txt', sep = '|')
slope = pd.read_csv('../../ALSFRS_slope_PROACT.txt', sep = '|')
slope = slope.dropna()

max_date = df[df.feature_name == 'ALSFRS_Total'][['feature_delta', 'SubjectID']]
max_date.loc[:, 'max_delta'] = max_date.feature_delta.astype(int)
max_date = max_date.groupby('SubjectID').max()
max_date = max_date[max_date.feature_delta >= 365][['max_delta']]

print df.shape, slope.shape, max_date.shape
slope_legal = pd.merge(slope, max_date, left_on="SubjectID", right_index=True)
slope_legal = slope_legal.drop('max_delta', 1)
data_with_slope = pd.merge(df, slope_legal, on="SubjectID")
print slope_legal.shape, data_with_slope.shape, data_with_slope.SubjectID.unique().size
data_with_slope.head()


# In[3]:

data = data_with_slope.drop(['ALSFRS_slope'], 1)
data.to_csv('../all_data.csv',sep='|', index=False)
slope_legal.to_csv('../all_slope.csv',sep='|', index=False)


# In[ ]:



